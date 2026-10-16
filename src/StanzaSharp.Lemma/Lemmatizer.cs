using System.IO.Compression;
using System.Text.Json;
using StanzaSharp.Nn;

namespace StanzaSharp.Lemma;

/// <summary>
/// Sets <see cref="Word.Lemma"/> from each word's text and UPOS. Port of Stanza's LemmaProcessor for an
/// <c>ensemble_dict</c> checkpoint (stanza/pipeline/lemma_processor.py, models/lemma/trainer.py):
/// a dictionary by (UPOS, word), then by word alone; words it misses go through a character seq2seq model
/// (models/common/seq2seq_model.py) with POS input, soft attention, a copy gate (not in older checkpoints: English craft,
/// genia, mimic) and an edit classifier (identity / lowercase / use the decoded string), decoded greedily. A <c>caseless</c>
/// model (English atis) sees the words lowercased, in the dictionary too; the edits and the fallback keep their case.
/// The network is an <see cref="ISeq2SeqNet"/> (per backend, shared with German MWT); the dictionary, DeltaVocab,
/// batching and the edits are here.
/// </summary>
internal sealed class Lemmatizer : IDisposable
{
    private const int UnkId = Seq2SeqConfig.UnkId, SosId = Seq2SeqConfig.SosId, EosId = Seq2SeqConfig.EosId;
    private const string Unk = "<UNK>";

    private readonly Dictionary<string, Dictionary<string, string?>> _posDict;
    private readonly Dictionary<string, int> _charToId, _posToId;
    private readonly string[] _idToChar;
    private readonly int _batchSize, _maxDecLen;
    private readonly bool _caseless;
    private readonly ISeq2SeqNet _net;

    /// <param name="net">Builds the network from the checkpoint and its sizes (a backend's).</param>
    internal Lemmatizer(Checkpoint ckpt, Func<Checkpoint, Seq2SeqConfig, ISeq2SeqNet> net)
    {
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
        Require(!Flag("charlm"), "a charlm");
        Require(config["num_layers"]?.GetValue<int>() == 1, "more than one encoder layer");
        Require(root["contextual"]?.AsArray().Count is null or 0, "contextual lemmatizers");
        Require(root["dicts_version"]?.GetValue<int>() == 3, "a dictionary format other than v3 (gzip + JSON)");

        // trainer._unpack_pos_dict: {pos: {word: lemma}}, "*" holding the POS-independent entries.
        using (LoadProfile.Measure(LoadProfile.Stage.Dict))
        {
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
            var raw = Convert.FromBase64String(root["dicts"]!["$bytes"]!.GetValue<string>());
            long t1 = System.Diagnostics.Stopwatch.GetTimestamp();
            var json = new MemoryStream();
            using (var gz = new GZipStream(new MemoryStream(raw), CompressionMode.Decompress))
                gz.CopyTo(json);
            long t2 = System.Diagnostics.Stopwatch.GetTimestamp();
            json.Position = 0;
            _posDict = Environment.GetEnvironmentVariable("STANZASHARP_DICT") is not ("reader" or "check")
                ? JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string?>>>(json)!
                : ReadDicts(json.GetBuffer().AsSpan(0, (int)json.Length));
            if (Environment.GetEnvironmentVariable("STANZASHARP_DICT") == "check")
            {
                json.Position = 0;
                var reference = JsonSerializer.Deserialize<Dictionary<string, Dictionary<string, string?>>>(json)!;
                if (reference.Count != _posDict.Count || reference.Any(p => !_posDict.TryGetValue(p.Key, out var d) || d.Count != p.Value.Count
                    || p.Value.Any(e => !d.TryGetValue(e.Key, out var v) || v != e.Value)))
                    throw new InvalidOperationException("ReadDicts differs from JsonSerializer");
                Console.Error.WriteLine("lemma dicts: ReadDicts equals JsonSerializer");
            }
            long t3 = System.Diagnostics.Stopwatch.GetTimestamp();
            if (LoadProfile.Enabled)
                Console.Error.WriteLine($"lemma dicts: base64 {raw.Length / 1024} KB gz {(t1 - t0) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F0} ms, gunzip {json.Length / 1024} KB {(t2 - t1) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F0} ms, deserialize {_posDict.Sum(d => d.Value.Count)} entries {(t3 - t2) * 1000.0 / System.Diagnostics.Stopwatch.Frequency:F0} ms");
        }

        var vocab = root["vocab"]!;
        _charToId = Checkpoint.UnitToId(vocab["char"]);
        _posToId = Checkpoint.UnitToId(vocab["pos"]);
        _idToChar = vocab["char"]!["_id2unit"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
        _batchSize = config["batch_size"]!.GetValue<int>();
        _maxDecLen = config["max_dec_len"]!.GetValue<int>();
        _caseless = Flag("caseless");
        var sizes = Seq2SeqConfig.From(config);
        _net = net(ckpt, sizes);
    }

    /// <summary>The dictionaries' JSON, <c>{pos: {word: lemma or null}}</c>, read like JsonSerializer (a repeated key: the last wins).</summary>
    private static Dictionary<string, Dictionary<string, string?>> ReadDicts(ReadOnlySpan<byte> json)
    {
        var reader = new Utf8JsonReader(json);
        var result = new Dictionary<string, Dictionary<string, string?>>();
        Expect(reader.Read() && reader.TokenType == JsonTokenType.StartObject);
        while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
        {
            var pos = reader.GetString()!;
            Expect(reader.Read() && reader.TokenType == JsonTokenType.StartObject);
            var words = new Dictionary<string, string?>();
            while (reader.Read() && reader.TokenType == JsonTokenType.PropertyName)
            {
                var word = reader.GetString()!;
                Expect(reader.Read() && reader.TokenType is JsonTokenType.String or JsonTokenType.Null);
                words[word] = reader.TokenType == JsonTokenType.Null ? null : reader.GetString();
            }

            Expect(reader.TokenType == JsonTokenType.EndObject);
            result[pos] = words;
        }

        Expect(reader.TokenType == JsonTokenType.EndObject);
        return result;

        static void Expect(bool ok)
        {
            if (!ok)
                throw new InvalidDataException("The lemma dictionaries are not {pos: {word: lemma}}");
        }
    }

    /// <summary>Loads <c>basePath.json</c> + <c>.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/lemma/combined_nocharlm</c>, on the managed backend.</summary>
    public static Lemmatizer LoadManaged(string basePath) => new(Checkpoint.Load(basePath), (ckpt, sizes) => new ManagedSeq2SeqNet(ckpt, sizes));

    /// <summary>Sets the lemma of every word in <paramref name="doc"/>; needs UPOS from the tagger.</summary>
    public void Process(Document doc, CancellationToken cancellationToken = default)
    {
        var words = doc.Sentences.SelectMany(s => s.Words).ToList();
        var lemmas = Lemmatize(words, cancellationToken);
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

    /// <summary>The text the model and the dictionary see: lowercased for a caseless model (data.lowercase_data).</summary>
    private string Input(Word word) => _caseless ? PyString.Lower(word.Text) : word.Text;

    /// <summary>trainer._lookup: the (UPOS, word) entry, then the POS-independent one.</summary>
    internal string? Lookup(string word, string? upos)
    {
        if (upos != null && _posDict.TryGetValue(upos, out var byPos) && byPos.GetValueOrDefault(word) is { } lemma)
            return lemma;
        return _posDict.TryGetValue("*", out var any) ? any.GetValueOrDefault(word) : null;
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

        using var decoder = _net.Encode(ids, batch, width, posIds, order.Select(i => (long)src[i].Length + 1).ToArray(), ct);
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

    public void Dispose() => _net.Dispose();
}
