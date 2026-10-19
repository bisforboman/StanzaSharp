using System.Text.Json.Nodes;

namespace StanzaSharp.Nn;

/// <summary>
/// A BERT-style transformer encoder's settings from its Hugging Face <c>config.json</c>: transformers' ElectraModel with
/// embedding_size = hidden_size (google/electra-large-discriminator, german-nlp-group's ELECTRA) or RobertaModel
/// (roberta-base), both with absolute positions, exact GELU and post-LayerNorm. Anything else throws, so a new transformer
/// can't load into the wrong maths.
/// </summary>
/// <param name="MaxPositions">The pieces a sentence can have: <c>max_position_embeddings</c> less <paramref name="PositionOffset"/>.</param>
/// <param name="Prefix">The encoder's prefix in <c>pytorch_model.bin</c>'s state dict: <c>electra.</c> (ElectraForPreTraining)
/// or <c>roberta.</c> (RobertaForMaskedLM).</param>
/// <param name="PositionOffset">The first position row a sentence uses: 0, or RoBERTa's <c>pad_token_id</c> + 1 (2), since
/// its create_position_ids_from_input_ids numbers the pieces from there.</param>
internal sealed record TransformerConfig(int Hidden, int Heads, int Layers, int Intermediate, int MaxPositions, double LayerNormEps, string Prefix = "electra.",
    int PositionOffset = 0)
{
    public static TransformerConfig Load(string configJson)
    {
        var c = JsonNode.Parse(File.ReadAllText(configJson)) ?? throw new InvalidDataException($"{configJson} is empty");
        string Text(string key) => c[key]?.GetValue<string>() ?? throw new InvalidDataException($"{configJson}: no {key}");
        int Int(string key) => c[key]?.GetValue<int>() ?? throw new InvalidDataException($"{configJson}: no {key}");

        // transformers' ElectraConfig and RobertaConfig default position_embedding_type to absolute (german-nlp-group's
        // config has none); RobertaConfig has no embedding_size (it is hidden_size).
        var type = Text("model_type");
        bool roberta = type == "roberta";
        if (type is not ("electra" or "roberta") || Text("hidden_act") != "gelu" || (c["position_embedding_type"]?.GetValue<string>() ?? "absolute") != "absolute"
            || (!roberta && Int("embedding_size") != Int("hidden_size")))
        {
            throw new InvalidDataException($"{configJson}: an architecture StanzaSharp doesn't support (ElectraModel with embedding_size = hidden_size, or RobertaModel; "
                + "exact GELU, absolute positions)");
        }

        int hidden = Int("hidden_size"), heads = Int("num_attention_heads");
        if (hidden % heads != 0)
            throw new InvalidDataException($"{configJson}: hidden_size {hidden} is not a multiple of num_attention_heads {heads}");
        int offset = roberta ? Int("pad_token_id") + 1 : 0;
        return new TransformerConfig(hidden, heads, Int("num_hidden_layers"), Int("intermediate_size"), Int("max_position_embeddings") - offset,
            c["layer_norm_eps"]?.GetValue<double>() ?? throw new InvalidDataException($"{configJson}: no layer_norm_eps"), roberta ? "roberta." : "electra.", offset);
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
    /// Whether <see cref="EncodeJoint"/> can encode this net's sentences together with those of the nets made from it by
    /// <see cref="WithLora"/> (the managed net); otherwise each net runs its own passes.
    /// </summary>
    bool EncodesJointly => false;

    /// <summary>
    /// <see cref="Encode"/> for several groups in one sweep over the layers (#169): each group is a net sharing this one's
    /// weights (this net or one of its <see cref="WithLora"/> nets) and its sentences, and gets exactly the states that net's
    /// own <see cref="Encode"/> gives them. A short call reads the weights once instead of once per net. Null if this net
    /// can't (<see cref="EncodesJointly"/> false, or a net of another transformer).
    /// </summary>
    float[][][]? EncodeJoint(IReadOnlyList<(ITransformerNet Net, IReadOnlyList<int[]> Sentences)> groups, int lastLayers, CancellationToken ct = default) => null;

    /// <summary>
    /// This encoder with <paramref name="lora"/>'s adapters added at run time. It shares this net's weights, so it must not
    /// outlive it; disposing it frees only the adapters.
    /// </summary>
    ITransformerNet WithLora(LoraWeights lora);
}
