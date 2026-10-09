#pragma warning disable CS0618 // DisableTf32 is obsolete (TorchSharp backend only) but still supported
using TorchSharp;

namespace StanzaSharp.Tests;

/// <summary>
/// Tests that change torch's process-wide settings (the TF32 switches) share this collection, so xUnit
/// never runs them in parallel with each other.
/// </summary>
[CollectionDefinition(Name)]
public sealed class TorchSettingsCollection
{
    public const string Name = "Torch process-wide settings";
}

[Collection(TorchSettingsCollection.Name)]
public class Tf32Tests
{
    [ModelFact]
    public void DisableTf32_TurnsOffBothSwitches()
    {
        var old = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32);
        try
        {
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = true;
            using var nlp = Pipeline.Load(Repo.Models, new PipelineOptions { Processors = "tokenize", DisableTf32 = true });
            Assert.False(torch.backends.cuda.matmul.allow_tf32);
            Assert.False(torch.backends.cudnn.allow_tf32);
        }
        finally
        {
            (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) = old;
        }
    }
}
