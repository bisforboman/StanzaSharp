namespace StanzaSharp;

/// <summary>
/// A model file of your own for one processor (<see cref="PipelineOptions.ModelPaths"/>), Stanza's
/// <c>{processor}_model_path</c>, with the shared files it reads (<c>{processor}_pretrain_path</c>,
/// <c>{processor}_forward_charlm_path</c>, <c>{processor}_backward_charlm_path</c>).
/// </summary>
/// <param name="Model">The processor's model file.</param>
/// <param name="Pretrain">The word vectors it reads; null: those of the model it replaces (none for a processor the
/// package lacks).</param>
/// <param name="ForwardCharlm">The forward character language model it reads; null: the replaced model's.</param>
/// <param name="BackwardCharlm">The backward character language model it reads; null: the replaced model's.</param>
/// <remarks>
/// Each path is a Stanza <c>.pt</c> file or a converted model's base path (the <c>.json</c> and <c>.safetensors</c> without
/// their extension), absolute or relative to the current directory. The model must be one StanzaSharp runs: a model
/// trained with Stanza's code for one of the architectures of the supported packages.
/// </remarks>
public sealed record ModelPath(string Model, string? Pretrain = null, string? ForwardCharlm = null, string? BackwardCharlm = null)
{
    /// <summary>A model file of your own; the shared files it reads are those of the model it replaces.</summary>
    public static implicit operator ModelPath(string model) => new(model);
}
