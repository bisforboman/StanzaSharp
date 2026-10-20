using System.Text.Json.Nodes;

namespace StanzaSharp.Nn;

/// <summary>
/// A model's LoRA adapters for its transformer (peft's state dict, a checkpoint's <c>bert_lora</c>: sentiment
/// <c>sstplus_electra-large</c>), applied at run time (owner's decision, 2026-10-17): each targeted linear layer computes
/// base(x) + B(A(x))·<see cref="Scaling"/>, as peft's LoRA Linear does in eval mode (dropout off).
/// </summary>
/// <param name="Weights">By module, e.g. <c>encoder.layer.0.attention.self.query</c>: A [rank, in] and B [out, rank], row-major.</param>
/// <param name="Scaling">lora_alpha / lora_rank.</param>
internal sealed record LoraWeights(IReadOnlyDictionary<string, (float[] A, float[] B, int Rank)> Weights, float Scaling)
{
    /// <summary>The modules peft's target names match in a BERT-style layer: query, value, and every <c>output.dense</c> and <c>intermediate.dense</c>.</summary>
    public static readonly string[] Modules = ["attention.self.query", "attention.self.value", "attention.output.dense", "intermediate.dense", "output.dense"];

    private const string PeftPrefix = "base_model.model.";

    /// <summary>Reads <paramref name="adapters"/> (<c>&lt;module&gt;.lora_A.weight</c>, <c>.lora_B.weight</c>) with the config's rank and alpha.</summary>
    public static LoraWeights Read(Checkpoint ckpt, JsonNode adapters, JsonNode config)
    {
        int rank = config["lora_rank"]?.GetValue<int>() ?? throw new InvalidDataException("No lora_rank");
        int alpha = config["lora_alpha"]?.GetValue<int>() ?? throw new InvalidDataException("No lora_alpha");
        var targets = config["lora_target_modules"]?.AsArray().Select(t => t!.GetValue<string>()).ToHashSet() ?? [];

        // A saved pooler (coref's modules_to_save) is never read: every model takes the hidden states, not the pooled output.
        var saved = config["lora_modules_to_save"]?.AsArray().Select(m => m!.GetValue<string>()).ToList() ?? [];
        if (saved.Any(m => m != "pooler"))
            throw new NotSupportedException($"LoRA modules_to_save {string.Join(", ", saved)} is not ported");
        var weights = new Dictionary<string, (float[], float[], int)>();
        foreach (var (rawKey, value) in adapters.AsObject())
        {
            // peft's own state dict names keys base_model.model.<module> (coref); the models' saved ones don't.
            var key = rawKey.StartsWith(PeftPrefix, StringComparison.Ordinal) ? rawKey[PeftPrefix.Length..] : rawKey;
            if (saved.Count > 0 && key.StartsWith("pooler.", StringComparison.Ordinal))
                continue;
            if (!key.EndsWith(".lora_A.weight", StringComparison.Ordinal))
            {
                if (!key.EndsWith(".lora_B.weight", StringComparison.Ordinal))
                    throw new NotSupportedException($"LoRA weight {key} is not ported");
                continue;
            }

            var module = key[..^".lora_A.weight".Length];
            if (!Modules.Any(m => module.EndsWith("." + m, StringComparison.Ordinal)) || !targets.Any(t => module.EndsWith(t, StringComparison.Ordinal)))
                throw new NotSupportedException($"LoRA module {module} is not one StanzaSharp applies");
            var a = ckpt.Tensor<float>(value);
            var b = ckpt.Tensor<float>(adapters[rawKey[..^".lora_A.weight".Length] + ".lora_B.weight"] ?? throw new InvalidDataException($"No {module}.lora_B.weight"));
            if (ckpt.Shape(value)[0] != rank)
                throw new InvalidDataException($"{key} has rank {ckpt.Shape(value)[0]}, not {rank}");
            weights[module] = (a, b, rank);
        }

        return new LoraWeights(weights, (float)alpha / rank);
    }

    /// <summary>Layer <paramref name="layer"/>'s adapter on <paramref name="module"/> (one of <see cref="Modules"/>), if any.</summary>
    public (float[] A, float[] B, int Rank)? Of(int layer, string module) =>
        Weights.TryGetValue($"encoder.layer.{layer}.{module}", out var w) ? w : null;
}
