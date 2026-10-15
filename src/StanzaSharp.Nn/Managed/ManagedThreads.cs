using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;

namespace StanzaSharp.Nn.Managed;

/// <summary>
/// Parallel regions for the managed kernels: a process-wide pool of <see cref="Count"/> − 1 worker threads that help
/// whichever calls have work, plus each calling thread itself (the managed twin of <c>torch.set_num_threads</c>).
/// </summary>
/// <remarks>
/// <para>An LSTM runs one region per time step, thousands per call, so a region must start in microseconds: idle
/// workers spin briefly before they block.</para>
/// <para>Concurrent callers don't take turns. Each caller always works on its own region, and the workers spread over
/// all open regions, so N callers use at most <see cref="Count"/> − 1 + N threads (libtorch gives each caller its own
/// team of <c>Threads</c>, N × Threads in all). With one caller this is a team of <see cref="Count"/>; with as many
/// callers as cores each call runs on about one core and none is starved.</para>
/// <para>Workers never make the busy threads (callers in a region plus awake workers) outnumber the logical
/// processors, or <see cref="Count"/> if that is larger: a region waits for its slowest item, so a worker preempted
/// while it holds one stalls its caller for a scheduler quantum. With more spinning threads than cores (8 callers on
/// 16 threads and 16 logical processors) single calls stalled for seconds (docs/performance/round-12-concurrent-calls-thread-pool.md).</para>
/// <para>Other processes can keep the cores busy too, which that count can't see. When the caller's wait for a region's
/// last item is far longer than an item (a worker holding one was preempted), one more worker stays asleep; after a
/// while without such a stall one comes back. Concurrent callers switch this off (docs/performance/round-32-pool-under-contention.md).</para>
/// <para>A region's items are claimed one at a time, so how they are split never depends on how many threads help:
/// results don't change with the thread count or with concurrent calls.</para>
/// </remarks>
internal static class ManagedThreads
{
    private static readonly Lock Gate = new();

    /// <summary>The default <see cref="Count"/>: <see cref="CpuCores.DefaultThreads"/> for this machine.</summary>
    public static readonly int Default = CpuCores.DefaultThreads(CpuCores.Physical(), Environment.ProcessorCount);

    private static int _count = Default;
    private static Pool? _pool;

    /// <summary>Threads per region, the caller included (at least 1).</summary>
    public static int Count
    {
        get => _count;
        set => _count = Math.Max(1, value);
    }

    /// <summary>The workers kept asleep after stalls (tests).</summary>
    internal static int Shed => _pool?.Shed ?? 0;

    /// <summary>Runs <paramref name="body"/>(i) for i in [0, n), on the calling thread and the pool's idle workers.</summary>
    // AggressiveOptimization here and on the pool's methods: an LSTM runs a region per time step, thousands per call,
    // and at JIT tier 0 (all of a fresh process's first calls) that machinery took most of a short call (round 17).
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void For(int n, Action<int> body)
    {
        if (n <= 1 || Count == 1)
        {
            for (int i = 0; i < n; i++)
                body(i);
            return;
        }

        var pool = _pool;
        if (pool?.Size != Count)
        {
            lock (Gate)
            {
                if (_pool?.Size != Count)
                {
                    _pool?.Stop();
                    _pool = new Pool(Count);
                }

                pool = _pool;
            }
        }

        pool.Run(n, body);
    }

    private sealed class Region(int n, Action<int> body)
    {
        public readonly int N = n;
        public readonly Action<int> Body = body;
        public int Next, Done;
        public Exception? Error;

        /// <summary>Claims and runs items until none is left. Returns how many it ran.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public int Drain()
        {
            int ran = 0;
            for (int i; (i = Interlocked.Increment(ref Next) - 1) < N;)
            {
                ran++;
                try
                {
                    Body(i);
                }
                catch (Exception e)
                {
                    Interlocked.CompareExchange(ref Error, e, null);
                }

                Interlocked.Increment(ref Done);
            }

            return ran;
        }
    }

    private sealed class Pool
    {
        private const int Slots = 64, SpinRounds = 2000;

        // A join wait counts as a stall when it is over StallTicks and over StallItems times the caller's mean item: a
        // preempted worker costs a scheduler quantum (ms), while an imbalance costs at most about one item. Every
        // WindowTicks, with k threads per region: stalls over 1/k of the window (more than one thread's worth) put one
        // more worker to sleep; under 1/(4k), one comes back.
        private const int StallItems = 8;
        private static readonly long StallTicks = Stopwatch.Frequency / 2000, WindowTicks = Stopwatch.Frequency / 10;

        private readonly Region?[] _open = new Region?[Slots];
        private readonly SemaphoreSlim _wake = new(0);
        private readonly int _limit;
        private volatile bool _stopped;

        // Callers inside Run plus workers not asleep; workers stay out (asleep) while this exceeds _limit.
        private int _sleepers, _busy;

        // Workers kept asleep after stalls (0 to _maxShed, which keeps max(1, Size / 4) workers): awake workers stay at
        // most Size − 1 − _shed. The window started at _windowAt and has seen _stalled ticks of stalls; _shared is 1 if
        // two callers overlapped in it.
        private readonly int _maxShed;
        private int _shed, _shared;
        private long _windowAt = Stopwatch.GetTimestamp(), _stalled;

        public int Size { get; }

        public int Shed => Volatile.Read(ref _shed);

        public Pool(int size)
        {
            Size = size;
            _limit = Math.Max(size, Environment.ProcessorCount);
            _busy = size - 1;
            _maxShed = Math.Max(0, size - 1 - Math.Max(1, size / 4));
            for (int i = 1; i < size; i++)
            {
                int id = i;
                new Thread(() => Work(id)) { IsBackground = true, Name = $"managed-kernel-{i}" }.Start();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int n, Action<int> body)
        {
            var region = new Region(n, body);
            int slot = -1;
            for (int s = 0; s < Slots && slot < 0; s++)
            {
                if (Interlocked.CompareExchange(ref _open[s], region, null) == null)
                    slot = s;
            }

            // ponytail: with more than 64 regions open at once the extra callers run alone; raise Slots if that happens.
            int busy = Interlocked.Increment(ref _busy), idleCores = _limit - busy;
            int asleep = Volatile.Read(ref _sleepers), shed = Volatile.Read(ref _shed);

            // Concurrent callers (busy threads − awake workers > 1) turn shedding off for the window: round 12's gate
            // already keeps our own threads within the cores, and shedding took help from the long calls among them.
            if (busy - (Size - 1 - asleep) > 1)
            {
                if (Volatile.Read(ref _shared) == 0)
                    Volatile.Write(ref _shared, 1);
                if (shed > 0)
                    Volatile.Write(ref _shed, shed = 0);
            }

            int sleepers = Math.Min(Math.Min(Math.Min(asleep, n - 1), idleCores), asleep - shed);
            if (slot >= 0 && sleepers > 0)
                _wake.Release(sleepers);
            long start = Stopwatch.GetTimestamp();
            int ran = region.Drain();
            if (slot >= 0)
                Volatile.Write(ref _open[slot], null);
            if (Volatile.Read(ref region.Done) < n)
            {
                long joined = Stopwatch.GetTimestamp();
                var spin = default(SpinWait);
                while (Volatile.Read(ref region.Done) < n)
                    spin.SpinOnce(-1);
                long wait = Stopwatch.GetTimestamp() - joined;
                if (ran > 0 && wait > StallTicks && wait > StallItems * (joined - start) / ran)
                    Interlocked.Add(ref _stalled, wait);
            }

            long window = Volatile.Read(ref _windowAt);
            if (start - window > WindowTicks && Interlocked.CompareExchange(ref _windowAt, start, window) == window)
                Adjust(shed, Interlocked.Exchange(ref _stalled, 0) * (1 - Interlocked.Exchange(ref _shared, 0)), start - window);

            Interlocked.Decrement(ref _busy);
            if (region.Error != null)
                ExceptionDispatchInfo.Throw(region.Error);
        }

        /// <summary>Ends a window of <paramref name="ticks"/> with <paramref name="stalled"/> ticks of stalls.</summary>
        private void Adjust(int shed, long stalled, long ticks)
        {
            int threads = Size - shed;
            if (stalled * threads > ticks && shed < _maxShed)
                Interlocked.CompareExchange(ref _shed, shed + 1, shed);
            else if (stalled * 4 * threads < ticks && shed > 0)
                Interlocked.CompareExchange(ref _shed, shed - 1, shed);
        }

        public void Stop()
        {
            _stopped = true;
            _wake.Release(Size);
        }

        /// <summary>Helps one open region with work left, scanning from a per-worker start so workers spread out.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private bool Help(int id)
        {
            for (int s = 0; s < Slots; s++)
            {
                if (Volatile.Read(ref _open[(s + id) % Slots]) is { } r && Volatile.Read(ref r.Next) < r.N)
                    return r.Drain() > 0;
            }

            return false;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private void Work(int id)
        {
            while (!_stopped)
            {
                bool found = false;
                for (int i = 0; i < SpinRounds && !found && !_stopped && Volatile.Read(ref _busy) <= _limit
                    && Volatile.Read(ref _sleepers) >= Volatile.Read(ref _shed); i++)
                {
                    if (!(found = Help(id)))
                        Thread.SpinWait(20);
                }

                if (found)
                    continue;

                // Announce the sleep before the last look: a caller that opens a region after that look sees us. _busy
                // drops first and rises last, so a caller counting callers as _busy minus the awake workers (Run) never
                // counts this worker as a caller, even if it is preempted between the two.
                Interlocked.Decrement(ref _busy);
                Interlocked.Increment(ref _sleepers);
                if (!_stopped && (Volatile.Read(ref _busy) >= _limit || !Help(id)))
                    _wake.Wait();
                Interlocked.Decrement(ref _sleepers);
                Interlocked.Increment(ref _busy);
            }
        }
    }
}
