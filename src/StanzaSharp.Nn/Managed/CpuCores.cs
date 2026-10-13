using System.Runtime.InteropServices;

namespace StanzaSharp.Nn.Managed;

/// <summary>The machine's physical cores, and the default thread count derived from them.</summary>
internal static class CpuCores
{
    /// <summary>The physical cores (all processor groups and sockets), or null if they can't be found. Never throws.</summary>
    public static int? Physical()
    {
        try
        {
            int? cores = OperatingSystem.IsWindows() ? Windows() : OperatingSystem.IsMacOS() ? MacOS() : OperatingSystem.IsLinux() ? Linux() : null;
            return cores > 0 ? cores : null;
        }
        catch (Exception)
        {
            return null;
        }
    }

    /// <summary>
    /// The default thread count: the physical cores when there are at least 4, else every logical processor (on 2 cores
    /// with SMT the 4 logical ones are faster; docs/performance/round-13-physical-cores.md); never more than <paramref name="logical"/>,
    /// which <see cref="Environment.ProcessorCount"/> limits to a container's CPU quota.
    /// </summary>
    public static int DefaultThreads(int? physical, int logical) => physical >= 4 ? Math.Min(physical.Value, logical) : logical;

    /// <summary>Distinct sibling sets of the online CPUs in sysfs: one per core.</summary>
    private static int? Linux() =>
        Directory.EnumerateDirectories("/sys/devices/system/cpu", "cpu*")
            .Select(cpu => Path.Combine(cpu, "topology", "thread_siblings_list"))
            .Where(File.Exists)
            .Select(f => File.ReadAllText(f).Trim())
            .Distinct()
            .Count();

    [DllImport("kernel32")]
    private static extern bool GetLogicalProcessorInformationEx(int relationship, IntPtr buffer, ref uint length);

    /// <summary>The RelationProcessorCore (0) records of GetLogicalProcessorInformationEx, one per core.</summary>
    private static int? Windows()
    {
        uint length = 0;
        GetLogicalProcessorInformationEx(0, IntPtr.Zero, ref length);
        if (length == 0)
            return null;
        var buffer = Marshal.AllocHGlobal((int)length);
        try
        {
            if (!GetLogicalProcessorInformationEx(0, buffer, ref length))
                return null;
            int cores = 0;
            for (int offset = 0, size; offset < length; offset += size, cores++)
            {
                size = Marshal.ReadInt32(buffer, offset + 4); // the record's Size follows its Relationship
                if (size <= 0)
                    return null;
            }

            return cores;
        }
        finally
        {
            Marshal.FreeHGlobal(buffer);
        }
    }

    [DllImport("libc")]
    private static extern int sysctlbyname(byte[] name, ref int value, ref nuint length, IntPtr newValue, nuint newLength);

    private static int? MacOS()
    {
        int cores = 0;
        nuint length = sizeof(int);
        return sysctlbyname("hw.physicalcpu\0"u8.ToArray(), ref cores, ref length, IntPtr.Zero, 0) == 0 ? cores : null;
    }
}
