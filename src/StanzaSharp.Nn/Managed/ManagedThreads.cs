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
/// <para>A region's items are claimed one at a time, so how they are split never depends on how many threads help:
/// results don't change with the thread count or with concurrent calls.</para>
/// </remarks>
internal static class ManagedThreads
{
    private static readonly Lock Gate = new();
    private static int _count = Environment.ProcessorCount;
    private static Pool? _pool;

    /// <summary>Threads per region, the caller included (at least 1).</summary>
    public static int Count
    {
        get => _count;
        set => _count = Math.Max(1, value);
    }

    /// <summary>Runs <paramref name="body"/>(i) for i in [0, n), on the calling thread and the pool's idle workers.</summary>
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
            lock (Gate)
            {
                if (_pool?.Size != Count)
                {
                    _pool?.Stop();
                    _pool = new Pool(Count);
                }
                pool = _pool;
            }
        pool.Run(n, body);
    }

    private sealed class Region(int n, Action<int> body)
    {
        public readonly int N = n;
        public readonly Action<int> Body = body;
        public int Next, Done;
        public Exception? Error;

        /// <summary>Claims and runs items until none is left. Returns whether it ran any.</summary>
        public bool Drain()
        {
            bool any = false;
            for (int i; (i = Interlocked.Increment(ref Next) - 1) < N;)
            {
                any = true;
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
            return any;
        }
    }

    private sealed class Pool
    {
        private const int Slots = 64, SpinRounds = 2000;
        private readonly Region?[] _open = new Region?[Slots];
        private readonly SemaphoreSlim _wake = new(0);
        private int _sleepers;
        private volatile bool _stopped;

        public int Size { get; }

        public Pool(int size)
        {
            Size = size;
            for (int i = 1; i < size; i++)
            {
                int id = i;
                new Thread(() => Work(id)) { IsBackground = true, Name = $"managed-kernel-{i}" }.Start();
            }
        }

        public void Run(int n, Action<int> body)
        {
            var region = new Region(n, body);
            int slot = -1;
            for (int s = 0; s < Slots && slot < 0; s++)
                if (Interlocked.CompareExchange(ref _open[s], region, null) == null)
                    slot = s;
            // ponytail: with more than 64 regions open at once the extra callers run alone; raise Slots if that happens.
            int sleepers = Math.Min(Volatile.Read(ref _sleepers), n - 1);
            if (slot >= 0 && sleepers > 0)
                _wake.Release(sleepers);
            region.Drain();
            if (slot >= 0)
                Volatile.Write(ref _open[slot], null);
            var spin = new SpinWait();
            while (Volatile.Read(ref region.Done) < n)
                spin.SpinOnce(-1);
            if (region.Error != null)
                ExceptionDispatchInfo.Throw(region.Error);
        }

        public void Stop()
        {
            _stopped = true;
            _wake.Release(Size);
        }

        /// <summary>Helps one open region with work left, scanning from a per-worker start so workers spread out.</summary>
        private bool Help(int id)
        {
            for (int s = 0; s < Slots; s++)
                if (Volatile.Read(ref _open[(s + id) % Slots]) is { } r && Volatile.Read(ref r.Next) < r.N)
                    return r.Drain();
            return false;
        }

        private void Work(int id)
        {
            while (!_stopped)
            {
                bool found = false;
                for (int i = 0; i < SpinRounds && !found && !_stopped; i++)
                    if (!(found = Help(id)))
                        Thread.SpinWait(20);
                if (found)
                    continue;
                // Announce the sleep before the last look: a caller that opens a region after that look sees us.
                Interlocked.Increment(ref _sleepers);
                if (!Help(id) && !_stopped)
                    _wake.Wait();
                Interlocked.Decrement(ref _sleepers);
            }
        }
    }
}
