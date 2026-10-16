using System.Diagnostics;

namespace StanzaSharp;

/// <summary>Round 34 instrumentation (prototype branch only): thread time per load stage, summed over threads.</summary>
internal static class LoadProfile
{
    public enum Stage { Parse, Tensor, Pack, Prep, Vocab, Dict, Gc, Model, CacheRead, CacheWrite }

    private static readonly long[] Ticks = new long[Enum.GetValues<Stage>().Length];

    [ThreadStatic]
    private static long[]? _local;

    public static bool Enabled { get; set; }

    public static long PackedFloats, TensorBytes;

    /// <summary>This thread's ticks per stage so far (a copy).</summary>
    public static long[] Local() => (long[])(_local ??= new long[Ticks.Length]).Clone();

    /// <summary>One line: this thread's ms per stage since <paramref name="before"/>.</summary>
    public static string Since(long[] before)
    {
        var now = Local();
        return string.Join(" ", Enum.GetValues<Stage>().Where(s => now[(int)s] != before[(int)s])
            .Select(s => $"{s}:{(now[(int)s] - before[(int)s]) * 1000.0 / Stopwatch.Frequency:F0}"));
    }

    public static Scope Measure(Stage stage) => new(stage);

    public static void Reset() => Array.Clear(Ticks);

    public static double Ms(Stage stage) => Interlocked.Read(ref Ticks[(int)stage]) * 1000.0 / Stopwatch.Frequency;

    public readonly struct Scope(Stage stage) : IDisposable
    {
        private readonly long _start = Enabled ? Stopwatch.GetTimestamp() : 0;

        public void Dispose()
        {
            if (_start != 0)
            {
                long t = Stopwatch.GetTimestamp() - _start;
                Interlocked.Add(ref Ticks[(int)stage], t);
                (_local ??= new long[Ticks.Length])[(int)stage] += t;
            }
        }
    }
}

/// <summary>Round 34 prototype: PackCache's hooks into <see cref="Checkpoint.Tensor{T}"/> on this thread.</summary>
internal static class TensorHooks
{
    /// <summary>Keys not to read (a cache hit): the tensor comes back uninitialized.</summary>
    [ThreadStatic]
    public static Func<string, bool>? Skip;

    /// <summary>Sees each tensor read (a cache miss).</summary>
    [ThreadStatic]
    public static Action<string, Array>? Read;
}
