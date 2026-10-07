using System.Runtime.InteropServices;

namespace StanzaSharp;

/// <summary>
/// Returns the native heap's free memory to the OS on Linux with glibc. libtorch allocates tensors with
/// <c>malloc</c>; once glibc has raised its mmap threshold (up to 32 MB) after the first large tensors are freed, later
/// tensors come from its heaps, and freed memory there stays in the process (RSS) unless it is at the top of a heap.
/// After a large <see cref="Pipeline.Process(string)"/> call that is hundreds of MB. <c>malloc_trim(0)</c> gives back the
/// free pages of every arena. It is a no-op elsewhere: Windows (libtorch uses its own mimalloc there), macOS and musl.
/// </summary>
internal static class NativeHeap
{
    private delegate int MallocTrimFunction(nuint pad);

    private static readonly MallocTrimFunction? MallocTrim = Find();

    /// <summary>Pipeline trims after calls of at least this many words (docs/performance.md, round 4).</summary>
    internal const int TrimMinWords = 1000;

    /// <summary>Whether <see cref="Trim"/> does anything here: Linux with glibc's <c>malloc_trim</c>.</summary>
    internal static bool CanTrim => MallocTrim != null;

    /// <summary>Gives the free pages of glibc's heaps back to the OS. Thread-safe; takes each arena's lock in turn.</summary>
    internal static void Trim() => MallocTrim?.Invoke(0);

    private static MallocTrimFunction? Find()
    {
        // libc.so.6 is glibc's soname; musl has neither it nor malloc_trim.
        if (!OperatingSystem.IsLinux() || !NativeLibrary.TryLoad("libc.so.6", out var libc) || !NativeLibrary.TryGetExport(libc, "malloc_trim", out var trim))
            return null;
        return Marshal.GetDelegateForFunctionPointer<MallocTrimFunction>(trim);
    }
}
