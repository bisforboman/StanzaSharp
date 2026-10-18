using System.Text.Json.Nodes;

namespace StanzaSharp.Nn;

/// <summary>
/// A BERT-style transformer encoder's settings from its Hugging Face <c>config.json</c>: the architecture
/// google/electra-large-discriminator uses (transformers' ElectraModel with embedding_size = hidden_size, absolute
/// positions, exact GELU, post-LayerNorm). Anything else throws, so a new transformer can't load into the wrong maths.
/// </summary>
internal sealed record TransformerConfig(int Hidden, int Heads, int Layers, int Intermediate, int MaxPositions, double LayerNormEps)
{
    public static TransformerConfig Load(string configJson)
    {
        var c = JsonNode.Parse(File.ReadAllText(configJson)) ?? throw new InvalidDataException($"{configJson} is empty");
        string Text(string key) => c[key]?.GetValue<string>() ?? throw new InvalidDataException($"{configJson}: no {key}");
        int Int(string key) => c[key]?.GetValue<int>() ?? throw new InvalidDataException($"{configJson}: no {key}");
        if (Text("model_type") != "electra" || Text("hidden_act") != "gelu" || Text("position_embedding_type") != "absolute"
            || Int("embedding_size") != Int("hidden_size"))
        {
            throw new InvalidDataException($"{configJson}: an architecture StanzaSharp doesn't support (ElectraModel with exact GELU, absolute positions, embedding_size = hidden_size)");
        }

        int hidden = Int("hidden_size"), heads = Int("num_attention_heads");
        if (hidden % heads != 0)
            throw new InvalidDataException($"{configJson}: hidden_size {hidden} is not a multiple of num_attention_heads {heads}");
        return new TransformerConfig(hidden, heads, Int("num_hidden_layers"), Int("intermediate_size"), Int("max_position_embeddings"),
            c["layer_norm_eps"]?.GetValue<double>() ?? throw new InvalidDataException($"{configJson}: no layer_norm_eps"));
    }
}

/// <summary>
/// The transformer encoder (docs/transformer-plan.md), arrays in and out like the other nets: <c>ManagedTransformer</c>
/// (Nn.Managed) and <c>TransformerNet</c> (StanzaSharp.Cuda). Sentences are id sequences of at most
/// <see cref="TransformerConfig.MaxPositions"/>, run without padding, so a sentence's states don't depend on the others in
/// the call (bitwise on the managed backend).
/// </summary>
internal interface ITransformerNet : IDisposable
{
    TransformerConfig Config { get; }

    /// <summary>
    /// Per sentence, the last <paramref name="lastLayers"/> hidden states (transformers' hidden_states[-lastLayers:]), as
    /// one array [lastLayers, pieces, Hidden], the earliest layer first.
    /// </summary>
    float[][] Encode(IReadOnlyList<int[]> sentences, int lastLayers, CancellationToken ct = default);

    /// <summary>
    /// Whether <see cref="Encode"/>'s arrays come from <see cref="System.Buffers.ArrayPool{T}.Shared"/> (and may be longer than
    /// they need to be). Their reader returns them: <see cref="TransformerEmbedder.Embed"/> once mixed, or the call's
    /// <see cref="TransformerCache"/> when it ends.
    /// </summary>
    bool RentsOutputs => false;

    /// <summary>
    /// This encoder with <paramref name="lora"/>'s adapters added at run time. It shares this net's weights, so it must not
    /// outlive it; disposing it frees only the adapters.
    /// </summary>
    ITransformerNet WithLora(LoraWeights lora);
}
