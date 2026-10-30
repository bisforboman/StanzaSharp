using System.Text.Json.Nodes;

namespace StanzaSharp.Nn;

/// <summary>Where a charlm checkpoint keeps its model (char_model.py's CharacterLanguageModel.load).</summary>
internal static class CharlmCheckpoint
{
    /// <summary>
    /// The model's full state (args, vocab, state_dict, is_forward_lm): the checkpoint's root, or, in a file the charlm
    /// trainer saved (Portuguese's oscar2023: model, epoch, global_step), its <c>model</c>.
    /// </summary>
    internal static JsonNode Model(Checkpoint ckpt) =>
        ckpt.Root["state_dict"] != null ? ckpt.Root : ckpt.Root["model"] ?? throw new InvalidDataException("Not a charlm checkpoint: no state_dict or model");
}
