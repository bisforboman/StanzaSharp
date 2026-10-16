using StanzaSharp.Nn;
using static TorchSharp.torch;

namespace StanzaSharp.Lemma;

/// <summary>Loads a <see cref="Lemmatizer"/> on either backend.</summary>
internal static class LemmatizerLoad
{
    extension(Lemmatizer)
    {
        /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/lemma/combined_nocharlm</c>.</summary>
        /// <param name="device">Where the model runs on TorchSharp; CPU by default.</param>
        /// <param name="backend">The network's backend; the managed one ignores <paramref name="device"/>.</param>
        /// <param name="forward">The forward charlm of a <c>_charlm</c> model (TorchSharp; the managed backend takes its own, <see cref="Lemmatizer.LoadManaged"/>).</param>
        /// <param name="backward">Its backward charlm.</param>
        public static Lemmatizer Load(string basePath, Device? device = null, Backend backend = Backend.TorchSharp, CharLanguageModel? forward = null,
            CharLanguageModel? backward = null) =>
            backend == Backend.Managed
                ? Lemmatizer.LoadManaged(basePath)
                : Weights.On(device, () => new Lemmatizer(Checkpoint.Load(basePath), (ckpt, sizes) => new Seq2SeqNet(ckpt, sizes), forward, backward));
    }
}
