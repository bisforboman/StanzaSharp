using System.Buffers;
using System.Text;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Ner;

/// <summary>
/// Named-entity recognizer: sets a BIOES tag on every token and builds each sentence's entities.
/// Port of stanza/models/ner/model.py <c>NERTagger</c> and trainer.py <c>predict</c> for the English and German
/// configurations:
/// - input is the pretrain vector plus a fine-tuned delta embedding if the model has one (both lowercased) and the
///   forward/backward charlm (<c>_charlm</c>) or the model's own bidirectional character LSTM's final
///   states (<c>_nocharlm</c>), through a linear <c>input_transform</c>
/// - a biLSTM, then one linear layer per tag set and CRF Viterbi decoding (the old format, e.g. German
///   <c>germeval2014</c>: one tag set, <c>tag_clf</c>/<c>crit</c>, which Stanza renames to <c>tag_clfs.0</c>/<c>crits.0</c>)
/// - <c>fix_singleton_tags</c> on the decoded sequence
/// NER reads tokens, not words, like Stanza: an MWT such as "don't" is tagged once.
/// </summary>
internal sealed class NerTagger : IDisposable
{
    internal const int UnkId = 1, PadId = 0; // vocab.UNK_ID / PAD_ID
    private const int VocabPrefixSize = 4;  // <PAD> <UNK> <EMPTY> <ROOT>

    private readonly Pretrain _pretrain;
    private readonly Dictionary<string, int>? _deltaVocab;
    private readonly bool _deltaSpaces, _lowercase, _charLowercase;
    private readonly string[] _tags;
    private readonly float[] _transitions;
    private readonly int _batchSize;
    private readonly INerNet _net;
    private readonly TransformerEmbedder? _transformer;
    private readonly float[]? _layerMix;

    /// <param name="transformer">The transformer a <c>_electra-large</c> model reads (the pipeline's, shared); else ignored.</param>
    /// <param name="tagSet">Stanza's <c>ner_predict_tagset</c>: which of the model's tag sets to decode (null: the checkpoint's
    /// <c>predict_tagset</c>, else 0).</param>
    internal NerTagger(Checkpoint ckpt, Pretrain pretrain, Func<int, int, bool, INerNet> net, TransformerEmbedder? transformer = null, int? tagSet = null)
    {
        _pretrain = pretrain;
        var config = ckpt.Root["config"]!;
        var vocab = ckpt.Root["vocab"]!;
        CheckSupported(config, vocab, pretrain);
        _layerMix = TransformerEmbedder.LayerMix(ckpt, transformer, "ner");
        _transformer = config["bert_model"] == null ? null : transformer;
        _deltaVocab = vocab["delta"] is { } delta ? Checkpoint.UnitToId(delta) : null;

        // The old format's delta (i2b2) is a PretrainedWordVocab, whose normalize_unit stores spaces as U+00A0.
        _deltaSpaces = vocab["_key2class"]?["delta"]?.GetValue<string>() == "PretrainedWordVocab";

        // lowercase: the word for both vocabs (off in the biomedical models); char_lowercase: the characters (radiology).
        _lowercase = config["lowercase"]?.GetValue<bool>() ?? true;
        _charLowercase = config["char_lowercase"]?.GetValue<bool>() == true;
        _batchSize = config["batch_size"]!.GetValue<int>();

        // The tag vocab has one column per tag set ("multi" NER: OntoNotes and WorldWide here). Stanza
        // decodes every column but keeps only predict_tagset's, so that is the only head we run.
        // The old format has one plain tag list and unnumbered heads (trainer.load renames them to column 0).
        int tagset = tagSet ?? config["predict_tagset"]?.GetValue<int>() ?? 0;
        var id2unit = vocab["tag"]!["_id2unit"]!;
        var columns = id2unit is JsonArray ? [0] : id2unit["$dict"]!.AsArray().Select(c => c![0]!.GetValue<int>()).ToList();
        if (!columns.Contains(tagset))
            throw new ArgumentException($"This NER model has {columns.Count} tag set{(columns.Count == 1 ? "" : "s")} ({string.Join(", ", columns)}), not {tagset}", nameof(tagSet));
        var tags = id2unit is JsonArray single ? single : id2unit["$dict"]!.AsArray().Single(c => c![0]!.GetValue<int>() == tagset)![1]!.AsArray();
        _tags = tags.Select(x => x!.GetValue<string>()).ToArray();
        _transitions = ckpt.Tensor<float>(ckpt.Root["model"]![HeadPrefix(ckpt, "crit", tagset) + "_transitions"]);
        _net = net(_tags.Length, tagset, NeedsCharlm(ckpt));
    }

    /// <summary><c>Load</c> (StanzaSharp.Cuda) on the managed backend (<see cref="Backend.Managed"/>), with the managed charlms.</summary>
    public static NerTagger LoadManaged(string basePath, Pretrain pretrain, ManagedCharLanguageModel? charlmForward, ManagedCharLanguageModel? charlmBackward,
        TransformerEmbedder? transformer = null, int? tagSet = null)
    {
        var ckpt = Checkpoint.Load(basePath);
        int transformerDim = TransformerDim(ckpt, transformer);
        return new NerTagger(ckpt, pretrain, (tags, tagset, charlm) =>
        {
            if (charlm && (charlmForward == null || charlmBackward == null || !charlmForward.IsForward || charlmBackward.IsForward))
                throw new ArgumentException("This NER model needs the forward charlm, then the backward one");
            return charlm
                ? new ManagedNerNet(ckpt, tags, tagset, pretrain, charlmForward, charlmBackward, transformerDim)
                : new ManagedNerNet(ckpt, tags, tagset, pretrain, null, null, transformerDim);
        }, transformer, tagSet);
    }

    /// <summary>The input columns a model gives <paramref name="transformer"/>'s vectors: its hidden size for a <c>_electra-large</c> model, else 0.</summary>
    internal static int TransformerDim(Checkpoint ckpt, TransformerEmbedder? transformer) =>
        ckpt.Root["config"]!["bert_model"] == null ? 0 : transformer?.Net.Config.Hidden ?? 0;

    /// <summary>Whether this model reads a transformer (<c>_electra-large</c>).</summary>
    internal bool UsesTransformer => _transformer != null;

    /// <summary>The sentences this tagger will send its transformer in <paramref name="doc"/>, for <see cref="TransformerEmbedder.Prefill"/> (null without one).</summary>
    internal TransformerReader? TransformerReader(Document doc) => _transformer == null
        ? null
        : new TransformerReader(_transformer, _layerMix, [.. doc.Sentences.Where(s => s.Tokens.Count > 0).Select(s => (IReadOnlyList<string>)[.. s.Tokens.Select(t => t.Text)])]);

    /// <summary>The delta vocabulary's size, or 0 for a model without a delta embedding (German <c>germeval2014</c>).</summary>
    internal static int DeltaCount(Checkpoint ckpt) => ckpt.Root["vocab"]!["delta"] is { } delta ? Checkpoint.UnitToId(delta).Count : 0;

    /// <summary>
    /// The prefix of tag set <paramref name="tagset"/>'s weights named <paramref name="name"/> (<c>tag_clf</c> or <c>crit</c>):
    /// <c>tag_clfs.N.</c>, or the old format's <c>tag_clf.</c>.
    /// </summary>
    internal static string HeadPrefix(Checkpoint ckpt, string name, int tagset) =>
        ckpt.Root["model"]![$"{name}s.{tagset}." + (name == "crit" ? "_transitions" : "weight")] != null ? $"{name}s.{tagset}." : $"{name}.";

    /// <summary>Sets <see cref="Token.Ner"/> on every token and rebuilds every sentence's <see cref="Sentence.Entities"/>.</summary>
    /// <param name="charlms">The tagger's charlm outputs, reused for sentences whose tokens are exactly its words (no MWT); each batch's entries are released after use, as NER is the last reader.</param>
    /// <param name="transformers">The call's transformer states: the tagger's, for sentences whose token texts are its words.</param>
    public void Process(Document doc, CharlmCache? charlms = null, CancellationToken cancellationToken = default, TransformerCache? transformers = null)
    {
        // ner/data.py: batches of batch_size sentences in document order.
        var sentences = doc.Sentences.Where(s => s.Tokens.Count > 0).ToList();
        foreach (var batch in sentences.Chunk(_batchSize))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var texts = batch.Select(s => (IReadOnlyList<string>)s.Tokens.Select(t => t.Text).ToList()).ToList();
            var keys = charlms == null
                ? null
                : batch.Select((s, i) => s.Words.Select(w => w.Text).SequenceEqual(texts[i]) ? s : null).ToList();
            var tags = Predict(texts, out _, charlms, keys, cancellationToken, keepEmissions: false, transformers);

            // NER runs last, so nothing reads these sentences' cached charlm outputs again.
            if (charlms != null)
            {
                foreach (var s in batch)
                    charlms.Release(s);
            }

            for (int i = 0; i < batch.Length; i++)
            {
                for (int j = 0; j < tags[i].Length; j++)
                    batch[i].Tokens[j].Ner = tags[i][j];
                Entity.Build(batch[i], doc.Text);
            }
        }
    }

    /// <summary>BIOES tags for each token, plus the emission scores (one [tokens, tags] array per sentence) for tests.</summary>
    /// <param name="charlms">With <paramref name="cacheKeys"/>: charlm outputs already computed; sentence i's are looked up
    /// under its key if not null.</param>
    /// <param name="keepEmissions">Fill <paramref name="emissions"/> (else it stays empty).</param>
    internal List<string[]> Predict(IReadOnlyList<IReadOnlyList<string>> sentences, out List<float[]> emissions,
        CharlmCache? charlms = null, IReadOnlyList<Sentence?>? cacheKeys = null, CancellationToken cancellationToken = default,
        bool keepEmissions = true, TransformerCache? transformers = null)
    {
        int batch = sentences.Count, width = sentences.Max(s => s.Count);

        // extract_bert_embeddings over the token texts as written, without the endpoints.
        var transformer = _transformer?.Embed(sentences, keepEndpoints: false, _layerMix, transformers, cancellationToken);

        // extract_static_embeddings: the word, lowercased if the model's lowercase is on, for both vocabs. A word the
        // pretrain knows but the delta vocab doesn't gets the delta's zero PAD row rather than its UNK row.
        var wordIds = new long[batch * width];
        var deltaIds = new long[batch * width];
        for (int i = 0; i < batch; i++)
        {
            for (int j = 0; j < sentences[i].Count; j++)
            {
                var key = _lowercase ? PyString.Lower(sentences[i][j]) : sentences[i][j];
                int word = _pretrain.UnitToId(key);
                wordIds[i * width + j] = word;
                if (_deltaVocab != null)
                {
                    int delta = _deltaVocab.GetValueOrDefault(_deltaSpaces ? key.Replace(' ', '\u00A0') : key, UnkId);
                    deltaIds[i * width + j] = delta == UnkId && word != _pretrain.UnkId ? PadId : delta;
                }
            }
        }

        // Rented (LOH-sized per batch): only the rows of each sentence's own tokens are written and read.
        // DataLoader.preprocess: with char_lowercase the character model or charlms read lowercased characters, which
        // differ from the tagger's charlm input, so nothing comes from or goes to the cache.
        if (_charLowercase)
        {
            sentences = [.. sentences.Select(s => (IReadOnlyList<string>)[.. s.Select(LowerChars)])];
            (charlms, cacheKeys) = (null, null);
        }

        int nTags = _tags.Length;
        var logits = ArrayPool<float>.Shared.Rent(batch * width * nTags);
        try
        {
            _net.Forward(sentences, wordIds, deltaIds, width, logits.AsSpan(0, batch * width * nTags), charlms, cacheKeys, cancellationToken, transformer);
            emissions = [];
            var result = new List<string[]>(batch);
            for (int i = 0; i < batch; i++)
            {
                var scores = logits.AsSpan(i * width * nTags, sentences[i].Count * nTags);
                if (keepEmissions)
                    emissions.Add(scores.ToArray());

                // The model can predict a tag below "O" (a vocab prefix); Stanza turns those into "O".
                var tags = Viterbi(scores, _transitions, nTags).Select(id => _tags[Math.Max(id, VocabPrefixSize)]).ToArray();
                result.Add(FixSingletonTags(tags));
            }

            return result;
        }
        finally
        {
            ArrayPool<float>.Shared.Return(logits);
        }
    }

    /// <summary>
    /// crf.py viterbi_decode in float32, like numpy: the highest-scoring tag sequence for
    /// <paramref name="scores"/> [n, tags] under <paramref name="transitions"/> [from, to]. Ties go to the
    /// lowest tag id, as with np.argmax.
    /// </summary>
    internal static int[] Viterbi(ReadOnlySpan<float> scores, float[] transitions, int tags)
    {
        int n = scores.Length / tags;

        // Rented: row 0 of the trellis is copied, every later row of both is written before it is read.
        var trellis = ArrayPool<float>.Shared.Rent(scores.Length);
        var backpointers = ArrayPool<int>.Shared.Rent(scores.Length);
        try
        {
            return Viterbi(scores, transitions, tags, n, trellis, backpointers);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(trellis);
            ArrayPool<int>.Shared.Return(backpointers);
        }
    }

    private static int[] Viterbi(ReadOnlySpan<float> scores, float[] transitions, int tags, int n, float[] trellis, int[] backpointers)
    {
        scores[..tags].CopyTo(trellis);
        for (int t = 1; t < n; t++)
        {
            for (int j = 0; j < tags; j++)
            {
                int best = 0;
                float max = trellis[(t - 1) * tags] + transitions[j];
                for (int i = 1; i < tags; i++)
                {
                    float v = trellis[(t - 1) * tags + i] + transitions[i * tags + j];
                    if (v > max)
                        (max, best) = (v, i);
                }

                trellis[t * tags + j] = scores[t * tags + j] + max;
                backpointers[t * tags + j] = best;
            }
        }

        var path = new int[n];
        int last = (n - 1) * tags;
        for (int j = 1; j < tags; j++)
        {
            if (trellis[last + j] > trellis[last + path[n - 1]])
                path[n - 1] = j;
        }

        for (int t = n - 1; t > 0; t--)
            path[t - 1] = backpointers[t * tags + path[t]];
        return path;
    }

    /// <summary>
    /// trainer.py fix_singleton_tags: an I- that cannot continue becomes E-, one that cannot start
    /// becomes B-; then a B- or E- standing alone becomes S-. Each check sees the tags already fixed
    /// to its left and the original ones to its right, as in Stanza.
    /// </summary>
    internal static string[] FixSingletonTags(string[] tags)
    {
        var fixedTags = (string[])tags.Clone();
        int n = fixedTags.Length;
        for (int i = 0; i < n; i++)
        {
            var tag = fixedTags[i];
            if (!tag.StartsWith("I-", StringComparison.Ordinal))
                continue;
            var type = tag[2..];
            if (i == n - 1 || (fixedTags[i + 1] != "I-" + type && fixedTags[i + 1] != "E-" + type))
                fixedTags[i] = "E-" + type;
            if (i == 0 || (fixedTags[i - 1] != "B-" + type && fixedTags[i - 1] != "I-" + type))
                fixedTags[i] = "B-" + type;
        }

        for (int i = 0; i < n; i++)
        {
            var tag = fixedTags[i];
            var type = tag.Length >= 2 ? tag[2..] : "";
            if (tag.StartsWith("B-", StringComparison.Ordinal)
                && (i == n - 1 || (fixedTags[i + 1] != "I-" + type && fixedTags[i + 1] != "E-" + type)))
            {
                fixedTags[i] = "S-" + type;
            }

            if (tag.StartsWith("E-", StringComparison.Ordinal)
                && (i == 0 || (fixedTags[i - 1] != "B-" + type && fixedTags[i - 1] != "I-" + type)))
            {
                fixedTags[i] = "S-" + type;
            }
        }

        return fixedTags;
    }

    private static void CheckSupported(JsonNode config, JsonNode vocab, Pretrain pretrain)
    {
        void Require(bool ok, string what)
        {
            if (!ok) throw new NotSupportedException($"NER checkpoint uses {what}, which is not ported");
        }

        Require(config["use_peft"]?.GetValue<bool>() != true, "a LoRA transformer");
        Require(config["char"]?.GetValue<bool>() == true && config["char_emb_dim"]?.GetValue<int>() > 0, "a configuration without character features");
        Require(config["word_emb_dim"]?.GetValue<int>() == pretrain.Dim, "a word embedding not matching the pretrain");
        Require(config["input_transform"]?.GetValue<bool>() == true, "a configuration without input_transform");
        Require(config["connect_output_layers"]?.GetValue<bool>() != true, "connect_output_layers");
        Require(vocab["word"] != null, "a vocab without word embeddings");

        // The word embedding is the pretrain's matrix (not saved in the checkpoint), indexed by this vocab.
        Require(vocab["word"]!["_id2unit"] is { } words && Checkpoint.Strings(words).Length == pretrain.Count, "a word vocab other than the pretrain's");

        // extract_static_embeddings applies the model's lowercase before either vocab's map; a delta vocab's own lower
        // flag would lowercase again, which only matters (and isn't ported) for a cased model.
        if (vocab["delta"] is { } delta)
        {
            Require(delta["ignore"]?.AsArray().Count is null or 0, "emb_finetune_known_only (a delta vocab with ignored words)");
            Require(config["lowercase"]?.GetValue<bool>() != false || delta["lower"]?.GetValue<bool>() != true, "a lowercasing delta vocab in a cased model");
        }
    }

    /// <summary>
    /// <c>char_lowercase</c>'s characters: Python lowercases each one on its own, and maps the result as one unit, so a
    /// character whose lowercase is longer (İ) becomes the unknown character.
    /// </summary>
    internal static string LowerChars(string word)
    {
        var lowered = new StringBuilder(word.Length);
        foreach (var rune in word.EnumerateRunes())
        {
            var lower = PyString.Lower(rune.ToString());
            lowered.Append(lower.EnumerateRunes().Skip(1).Any() ? "\uFFFF" : lower);
        }

        return lowered.ToString();
    }

    /// <summary>Whether a checkpoint reads the shared charlms (<c>_charlm</c>) rather than its own character model (<c>_nocharlm</c>).</summary>
    private static bool NeedsCharlm(Checkpoint ckpt) => ckpt.Root["config"]!["charlm"]?.GetValue<bool>() == true;

    public void Dispose() => _net.Dispose();
}
