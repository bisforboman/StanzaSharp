using System.IO.Compression;
using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Lemma;

/// <summary>
/// Sets <see cref="Word.Lemma"/> from each word's text and UPOS. Port of Stanza's LemmaProcessor for an
/// <c>ensemble_dict</c> checkpoint (stanza/pipeline/lemma_processor.py, models/lemma/trainer.py):
/// a dictionary by (UPOS, word), then by word alone; words it misses go through a character seq2seq model
/// (models/common/seq2seq_model.py) with POS input, soft attention, a copy gate (not in older checkpoints: English craft,
/// genia, mimic) and an edit classifier (identity / lowercase / use the decoded string), decoded greedily. A <c>_charlm</c>
/// model also feeds the encoder both charlms' states over <c>"\n" + word + " "</c> (seq2seq_model.py embed). A <c>caseless</c>
/// model (English atis) sees the words lowercased, in the dictionary too; the edits and the fallback keep their case.
/// The network is an <see cref="ISeq2SeqNet"/> (per backend, shared with German MWT); the dictionary, DeltaVocab,
/// batching and the edits are here. A checkpoint's <c>contextual</c> models (<see cref="LemmaClassifier"/>, English
/// <c>combined_charlm</c>) then replace the lemmas of their target words.
/// </summary>
internal sealed class Lemmatizer : IDisposable
{
    private const int UnkId = Seq2SeqConfig.UnkId, SosId = Seq2SeqConfig.SosId, EosId = Seq2SeqConfig.EosId;
    private const string Unk = "<UNK>";

    private readonly LemmaDictionary _dict;
    private readonly Dictionary<string, int> _charToId, _posToId;
    private readonly string[] _idToChar;
    private readonly int _maxDecLen;
    private int _batchSize; // set once: the constructor or WithBatchSize's copy
    private readonly bool _caseless, _dictOnly;
    private readonly ISeq2SeqNet _net;
    private readonly IPerCharLanguageModel? _forward, _backward; // a _charlm model's charlms
    private readonly LemmaClassifier[] _contextual;

    /// <param name="net">Builds the network from the checkpoint and its sizes (a backend's).</param>
    /// <param name="forward">The forward charlm a <c>_charlm</c> model reads (config <c>charlm</c>); else null.</param>
    /// <param name="backward">Its backward charlm.</param>
    /// <param name="pretrain">The pretrain the checkpoint's contextual classifiers read (resources.json's dependency); else null.</param>
    /// <param name="classifierNet">Builds a contextual classifier's network from the checkpoint and its state dict.</param>
    /// <param name="dictOnly">Stanza's <c>lemma_dict_only</c>: the dictionary alone, the word where it has no entry; neither
    /// the seq2seq model nor the contextual classifiers run.</param>
    internal Lemmatizer(Checkpoint ckpt, Func<Checkpoint, Seq2SeqConfig, ISeq2SeqNet> net, IPerCharLanguageModel? forward = null,
        IPerCharLanguageModel? backward = null, Pretrain? pretrain = null, Func<Checkpoint, JsonNode, ILemmaClassifierNet>? classifierNet = null,
        bool dictOnly = false)
    {
        _dictOnly = dictOnly;
        var root = ckpt.Root;
        var config = root["config"]!;
        bool Flag(string key) => config[key]?.GetValue<bool>() == true;
        void Require(bool ok, string what)
        {
            if (!ok)
                throw new NotSupportedException($"Lemmatizer checkpoints with {what} are not ported");
        }

        Require(!Flag("dict_only") && Flag("ensemble_dict"), "dict_only or without ensemble_dict");
        Require(config["beam_size"]?.GetValue<int>() == 1, "beam search (beam_size > 1)");
        Require(config["attn_type"]?.GetValue<string>() == "soft", $"attn_type {config["attn_type"]}");
        Require(Flag("edit") && config["num_edit"]?.GetValue<int>() == 3, "edit off or num_edit != 3");
        Require(Flag("pos") && config["pos_dim"]?.GetValue<int>() > 0, "POS input off");
        Require(config["num_layers"]?.GetValue<int>() == 1, "more than one encoder layer");
        Require(root["dicts_version"]?.GetValue<int>() == 3, "a dictionary format other than v3 (gzip + JSON)");

        // trainer._unpack_pos_dict: {pos: {word: lemma}}, "*" holding the POS-independent entries.
        var json = new MemoryStream();
        using (var gz = new GZipStream(new MemoryStream(Convert.FromBase64String(root["dicts"]!["$bytes"]!.GetValue<string>())), CompressionMode.Decompress))
            gz.CopyTo(json);
        _dict = new LemmaDictionary(json.GetBuffer(), (int)json.Length);

        var vocab = root["vocab"]!;
        _charToId = Checkpoint.UnitToId(vocab["char"]);
        _posToId = Checkpoint.UnitToId(vocab["pos"]);
        _idToChar = vocab["char"]!["_id2unit"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
        _batchSize = config["batch_size"]!.GetValue<int>();
        _maxDecLen = config["max_dec_len"]!.GetValue<int>();
        _caseless = Flag("caseless");

        // trainer.build_seq2seq: with charlm, CharacterLanguageModelWordAdapter([forward, backward]) after the embedding.
        // Charlms given to a model without them are ignored, as the tagger and the parsers do (PipelineOptions.ModelPaths).
        int contextual = 0;
        if (Flag("charlm"))
        {
            if (forward == null || backward == null)
                throw new ArgumentException("This lemmatizer reads a forward and a backward charlm; give both", nameof(forward));
            contextual = forward.HiddenDim + backward.HiddenDim;
            int encoderInput = (int)ckpt.Shape(root["model"]!["encoder.weight_ih_l0"])[1];
            if (encoderInput != config["emb_dim"]!.GetValue<int>() + contextual)
                throw new InvalidDataException($"The encoder reads {encoderInput} inputs, not the embedding + both charlms' {contextual}");
            (_forward, _backward) = (forward, backward);
        }

        var sizes = Seq2SeqConfig.From(config, contextual);
        _net = net(ckpt, sizes);

        // trainer.load: LemmaClassifier.from_checkpoint for each, reading the pipeline's pretrain and charlms.
        var classifiers = root["contextual"]?.AsArray() ?? [];
        if (classifiers.Count > 0 && (pretrain == null || classifierNet == null))
            throw new ArgumentException("This lemmatizer's contextual classifiers read the pretrain; give it", nameof(pretrain));
        _contextual = [.. classifiers.Select(node => new LemmaClassifier(ckpt, node!, pretrain!, classifierNet!))];
    }

    /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/lemma/combined_nocharlm</c>, on the managed backend.</summary>
    /// <param name="forward">The forward charlm of a <c>_charlm</c> model; null for the others.</param>
    /// <param name="backward">Its backward charlm.</param>
    /// <param name="pretrain">The pretrain of a model with contextual classifiers (English <c>combined_charlm</c>); null for the others.</param>
    /// <param name="dictOnly">Stanza's <c>lemma_dict_only</c>: the dictionary alone.</param>
    public static Lemmatizer LoadManaged(string basePath, ManagedCharLanguageModel? forward = null, ManagedCharLanguageModel? backward = null,
        Pretrain? pretrain = null, bool dictOnly = false) =>
        new(Checkpoint.Load(basePath), (ckpt, sizes) => new ManagedSeq2SeqNet(ckpt, sizes), forward, backward, pretrain,
            (ckpt, model) => new ManagedLemmaClassifierNet(ckpt, model, pretrain!,
                forward ?? throw new ArgumentException("The contextual classifiers read both charlms", nameof(forward)),
                backward ?? throw new ArgumentException("The contextual classifiers read both charlms", nameof(backward))), dictOnly);

    /// <summary>Sets the lemma of every word in <paramref name="doc"/>; needs UPOS from the tagger.</summary>
    /// <param name="onContextual">For tests: each contextual classifier's labels and [targets, labels] logits, in Stanza's order.</param>
    public void Process(Document doc, CancellationToken cancellationToken = default, Action<string[], float[]>? onContextual = null)
    {
        var words = doc.Sentences.SelectMany(s => s.Words).ToList();
        var lemmas = _dictOnly ? LemmatizeByDictionary(words) : Lemmatize(words, cancellationToken);

        // trainer.update_contextual_preds, reversed so that the first classifier has priority (not with dict_only).
        if (_contextual.Length > 0 && !_dictOnly)
        {
            var sentences = doc.Sentences.Select(s => (IReadOnlyList<Word>)[.. s.Words]).ToList();
            for (int i = _contextual.Length - 1; i >= 0; i--)
            {
                cancellationToken.ThrowIfCancellationRequested();
                _contextual[i].Apply(sentences, lemmas, cancellationToken, onContextual);
            }
        }

        for (int i = 0; i < words.Count; i++)

            // Stanza's Word.lemma setter stores "_" as None unless the word itself is "_".
            words[i].Lemma = lemmas[i] == "_" && words[i].Text != "_" ? null : lemmas[i];
    }

    /// <summary>LemmaProcessor with <c>use_identity</c> (lemma model "identity", e.g. English eslspok): each word's lemma is its text.</summary>
    public static void Identity(Document doc)
    {
        foreach (var word in doc.Sentences.SelectMany(s => s.Words))
            word.Lemma = word.Text.Length == 0 ? null : word.Text; // "" becomes "_", which the Word.lemma setter stores as None
    }

    /// <summary>LemmaProcessor.process: dictionary hits, else the seq2seq prediction; "_" for an empty lemma.</summary>
    internal List<string> Lemmatize(IReadOnlyList<Word> words, CancellationToken cancellationToken = default)
    {
        var lemmas = words.Select(w => Lookup(Input(w), w.Upos)).ToList();
        var misses = Enumerable.Range(0, words.Count).Where(i => lemmas[i] == null).ToList();
        var predicted = Postprocess(misses.Select(i => words[i]).ToList(), cancellationToken);
        for (int k = 0; k < misses.Count; k++)
            lemmas[misses[k]] = predicted[k];
        return lemmas.Select(l => l!.Length == 0 ? "_" : l).ToList();
    }

    /// <summary>trainer.predict_dict (<c>lemma_dict_only</c>): the dictionary's lemma, else the word as the dictionary saw it; "_" for an empty one.</summary>
    internal List<string> LemmatizeByDictionary(IReadOnlyList<Word> words) =>
        [.. words.Select(w => Lookup(Input(w), w.Upos) ?? Input(w)).Select(l => l.Length == 0 ? "_" : l)];

    /// <summary>The text the model and the dictionary see: lowercased for a caseless model (data.lowercase_data).</summary>
    private string Input(Word word) => _caseless ? PyString.Lower(word.Text) : word.Text;

    /// <summary>trainer._lookup: the (UPOS, word) entry, then the POS-independent one.</summary>
    internal string? Lookup(string word, string? upos)
    {
        return (upos == null ? null : _dict.Lookup(upos, word)) ?? _dict.Lookup("*", word);
    }

    /// <summary>trainer.postprocess over the seq2seq output: apply the edit, and fall back to the word on empty or &lt;UNK&gt;.</summary>
    internal List<string> Postprocess(IReadOnlyList<Word> words, CancellationToken cancellationToken = default)
    {
        var (decoded, edits) = Predict(words, cancellationToken);
        var result = new List<string>(words.Count);
        for (int i = 0; i < words.Count; i++)
        {
            var word = words[i].Text;
            var lemma = edits[i] switch { 1 => word, 2 => PyString.Lower(word), _ => decoded[i] };
            result.Add(lemma.Length == 0 || lemma.Contains(Unk, StringComparison.Ordinal) ? word : lemma);
        }

        return result;
    }

    /// <summary>
    /// The seq2seq model on the words in batches of <c>batch_size</c>, as Stanza's DataLoader feeds it:
    /// the decoded string (before edits) and the edit class of each word.
    /// </summary>
    /// <param name="onStep">For tests: called after each decoder step with its log-probs, the row width and which rows were done before it.</param>
    /// <param name="onEdit">For tests: called at the end of each batch with its [batch, 3] edit logits (rows sorted like the steps').</param>
    internal (List<string> Decoded, List<int> Edits) Predict(IReadOnlyList<Word> words, CancellationToken cancellationToken = default,
        Action<float[], int, bool[]>? onStep = null, Action<float[]>? onEdit = null)
    {
        // DeltaVocab: characters the vocabulary lacks get new ids past it, in code point order, so the copy
        // gate can still output them. Stanza builds it over the text, UPOS and current lemma of every word.
        var charToId = _charToId;
        var idToChar = _idToChar;
        var unknown = words.SelectMany(w => (Input(w) + (w.Upos ?? "_") + (w.Lemma ?? "_")).EnumerateRunes())
            .Where(r => !_charToId.ContainsKey(r.ToString()))
            .Distinct()
            .Order()
            .Select(r => r.ToString())
            .ToList();
        if (unknown.Count > 0)
        {
            charToId = new Dictionary<string, int>(_charToId);
            foreach (var c in unknown)
                charToId[c] = charToId.Count;
            idToChar = [.. _idToChar, .. unknown];
        }

        var decoded = new List<string>(words.Count);
        var edits = new List<int>(words.Count);
        for (int start = 0; start < words.Count; start += _batchSize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var batch = words.Skip(start).Take(_batchSize).ToList();
            var (d, e) = PredictBatch(batch, charToId, idToChar, cancellationToken, onStep, onEdit);
            decoded.AddRange(d);
            edits.AddRange(e);
        }

        return (decoded, edits);
    }

    private (string[] Decoded, int[] Edits) PredictBatch(List<Word> words, Dictionary<string, int> charToId, string[] idToChar, CancellationToken ct,
        Action<float[], int, bool[]>? onStep, Action<float[]>? onEdit)
    {
        // <SOS> chars <EOS>, one unit per code point; sorted like data.sort_all: longest first, ties by later index first.
        var src = words.Select(w => Input(w).EnumerateRunes().Select(r => charToId[r.ToString()]).Prepend(SosId).Append(EosId).ToArray()).ToList();
        var order = Enumerable.Range(0, words.Count).OrderByDescending(i => src[i].Length).ThenByDescending(i => i).ToArray();
        int batch = words.Count, width = src[order[0]].Length;
        var ids = new long[batch * width];
        for (int r = 0; r < batch; r++)
        {
            for (int k = 0; k < src[order[r]].Length; k++)
                ids[r * width + k] = src[order[r]][k];
        }

        var posIds = order.Select(i => (long)_posToId.GetValueOrDefault(words[i].Upos ?? "_", UnkId)).ToArray();

        var contextual = Contextual(words, order, width + 1, ct);
        using var decoder = _net.Encode(ids, batch, width, posIds, order.Select(i => (long)src[i].Length + 1).ToArray(), ct, contextual: contextual);
        var output = Seq2SeqConfig.Greedy(decoder, batch, _maxDecLen, neverDecodeUnk: false, ct, onStep);

        var editValues = decoder.EditLogits;
        onEdit?.Invoke(editValues);
        var decoded = new string[batch];
        var edits = new int[batch];
        for (int r = 0; r < batch; r++)
        {
            decoded[order[r]] = string.Concat(output[r].Select(id => idToChar[id]));
            int best = 0; // numpy argmax: the first maximum
            for (int k = 1; k < 3; k++)
            {
                if (editValues[r * 3 + k] > editValues[r * 3 + best])
                    best = k;
            }

            edits[order[r]] = best;
        }

        return (decoded, edits);
    }

    /// <summary>
    /// A <c>_charlm</c> model's encoder input after the embedding (seq2seq_model.py embed): per sorted row r and encoder step t
    /// (of <paramref name="steps"/>), the forward then the backward charlm's state after unit t of <c>"\n" + word + " "</c>
    /// (both left to right: per_char_representation). Its n + 2 states fill steps 0 to n + 1, so the UPOS step gets the
    /// state after "\n", and the last (the word's <c>&lt;EOS&gt;</c>, Stanza's appended raw_zeros row) zeros. Null without charlms.
    /// </summary>
    private float[]? Contextual(List<Word> words, int[] order, int steps, CancellationToken ct)
    {
        if (_forward == null || _backward == null)
            return null;
        int f = _forward.HiddenDim, dim = f + _backward.HiddenDim;
        var forward = new int[order.Length][];
        var backward = new int[order.Length][];
        var firstRow = new int[order.Length];
        for (int r = 0; r < order.Length; r++)
        {
            var units = Input(words[order[r]]).EnumerateRunes().Select(rune => rune.ToString()).Prepend("\n").Append(" ").ToArray();
            forward[r] = [.. units.Select(u => _forward.Id(u))];
            backward[r] = [.. units.Select(u => _backward.Id(u))];
            firstRow[r] = r * steps;
        }

        var contextual = new float[order.Length * steps * dim];
        _forward.PerChar(forward, firstRow, contextual, dim, ct);
        _backward.PerChar(backward, firstRow, contextual.AsSpan(f), dim, ct);
        return contextual;
    }

    /// <summary>
    /// A copy batching <paramref name="batchSize"/> words per seq2seq batch (Stanza's <c>lemma_batch_size</c>). It
    /// shares the networks (the pipeline keeps only the copy), so it is made at Load, before any call.
    /// </summary>
    internal Lemmatizer WithBatchSize(int batchSize)
    {
        var copy = (Lemmatizer)MemberwiseClone();
        copy._batchSize = batchSize;
        return copy;
    }

    public void Dispose()
    {
        _net.Dispose();
        foreach (var classifier in _contextual)
            classifier.Dispose();
    }
}
