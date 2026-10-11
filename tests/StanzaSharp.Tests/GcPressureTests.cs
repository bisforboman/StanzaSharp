using StanzaSharp;
using Xunit;

namespace StanzaSharp.Tests;

/// <summary>
/// TorchSharp frees a temporary Scalar (e.g. the alpha of every tensor add) in its finalizer, and nothing keeps it
/// alive during the native call that reads it. Collections started by other threads while Process is inside libtorch
/// can then free it mid-call. This runs Process while another thread collects and finalizes nonstop.
/// </summary>
[Trait("Backend", "TorchSharp")] // TorchSharp-only: not in a ready pull request's light golden run (ci.yml)
public class GcPressureTests
{
    [ModelTheory]
    [InlineData("default")]
    [InlineData("default_fast")]
    public void Process_UnderConstantCollections_GivesTheSameOutput(string package)
    {
        using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Package = package, Backend = CudaBackend.Cpu });
        var corpus = File.ReadAllText(Path.Combine(Repo.Golden, "corpus.txt"));
        var expected = Conllu.Write(nlp.Process(corpus));
        using var stop = new CancellationTokenSource();
        var collector = new Thread(() =>
        {
            while (!stop.IsCancellationRequested)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        });
        collector.Start();
        try
        {
            for (int i = 0; i < 3; i++)
                Assert.Equal(expected, Conllu.Write(nlp.Process(corpus)));
        }
        finally
        {
            stop.Cancel();
            collector.Join();
        }
    }
}
