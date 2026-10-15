using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using StanzaSharp.Nn;

namespace StanzaSharp.Mwt;

/// <summary>
/// Expands tokens the tokenizer marked as multi-word (<see cref="Token.IsMwtCandidate"/>) into
/// words, e.g. "don't" → "do" + "n't". Port of Stanza's MWTProcessor (stanza/models/mwt/trainer.py): a dictionary
/// first (<c>ensemble_dict</c>), then the model, either
/// - a character classifier that predicts where to cut the token (<c>force_exact_pieces</c>, English:
///   stanza/models/mwt/character_classifier.py), or
/// - a character seq2seq model that writes the expansion (German, English partut: models/common/seq2seq_model.py, shared
///   with the lemmatizer as <see cref="ISeq2SeqNet"/>), decoded by a beam search of 5 without <c>&lt;UNK&gt;</c>.
/// </summary>
internal sealed class MwtExpander : IDisposable
{
    private readonly Dictionary<string, string> _dict;
    private readonly Dictionary<string, int> _vocab;
    private readonly bool _useDict, _useModel;
    private readonly int _padId, _unkId, _sosId, _eosId, _batchSize, _maxDecLen;

    // trainer.predict passes its beam_size (1) where Seq2SeqModel.predict takes pos, so Stanza always searches the default 5 beams.
    private const int BeamSize = 5;
    private readonly string[] _idToChar;
    private readonly IMwtNet? _net;
    private readonly ISeq2SeqNet? _seq2seq;

    /// <param name="net">Builds the classifier from the checkpoint, its config and the PAD id (a backend's).</param>
    /// <param name="seq2seq">Builds the seq2seq network from the checkpoint and its sizes (a backend's).</param>
    internal MwtExpander(Checkpoint ckpt, Func<Checkpoint, JsonNode, int, IMwtNet> net, Func<Checkpoint, Seq2SeqConfig, ISeq2SeqNet> seq2seq)
    {
        var config = ckpt.Root["config"]!;
        bool dictOnly = config["dict_only"]!.GetValue<bool>();
        _useModel = !dictOnly;
        _useDict = dictOnly || config["ensemble_dict"]?.GetValue<bool>() == true;
        _dict = ckpt.Root["dict"]!.AsObject().ToDictionary(kv => kv.Key, kv => kv.Value!.GetValue<string>());
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        (_padId, _unkId, _sosId, _eosId) = (_vocab["<PAD>"], _vocab["<UNK>"], _vocab["<SOS>"], _vocab["<EOS>"]);
        _idToChar = ckpt.Root["vocab"]!["_id2unit"]!.AsArray().Select(n => n!.GetValue<string>()).ToArray();
        if (!_useModel)
            return;
        if (config["force_exact_pieces"]?.GetValue<bool>() == true)
        {
            _net = net(ckpt, config, _padId);
            return;
        }

        void Require(bool ok, string what)
        {
            if (!ok)
                throw new NotSupportedException($"MWT seq2seq checkpoints with {what} are not ported");
        }

        Require(config["beam_size"]?.GetValue<int>() == 1, "beam search (beam_size > 1)");
        Require(config["attn_type"]?.GetValue<string>() == "soft", $"attn_type {config["attn_type"]}");
        Require(config["copy"]?.GetValue<bool>() == true, "copy off");
        Require(config["num_layers"]?.GetValue<int>() == 1, "more than one encoder layer");
        Require((_padId, _unkId, _sosId, _eosId) == (Seq2SeqConfig.PadId, Seq2SeqConfig.UnkId, Seq2SeqConfig.SosId, Seq2SeqConfig.EosId),
            "special ids other than seq2seq_constant's");
        _batchSize = config["batch_size"]!.GetValue<int>();
        _maxDecLen = config["max_dec_len"]!.GetValue<int>();
        var sizes = Seq2SeqConfig.From(config);
        Require(!sizes.UsesPos && !sizes.Edit, "POS input or an edit classifier");
        _seq2seq = seq2seq(ckpt, sizes);
    }

    /// <summary>Loads <c>basePath.json</c> + <c>basePath.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/mwt/combined</c>, on the managed backend.</summary>
    public static MwtExpander LoadManaged(string basePath) =>
        new(Checkpoint.Load(basePath), (ckpt, config, _) => new ManagedMwtNet(ckpt, config), (ckpt, sizes) => new ManagedSeq2SeqNet(ckpt, sizes));

    /// <summary>Expands marked tokens in place and renumbers each sentence's words.</summary>
    public void Process(Document doc)
    {
        var candidates = doc.Sentences.SelectMany(s => s.Tokens).Where(t => t.IsMwtCandidate).ToList();
        var expansions = Expand(candidates.Select(t => t.Text).ToList());
        for (int i = 0; i < candidates.Count; i++)
            SetWords(candidates[i], expansions[i]);

        foreach (var sent in doc.Sentences)
        {
            int id = 1;
            foreach (var word in sent.Words)
                word.Id = id++;
        }
    }

    /// <summary>Space-separated expansion for each token text.</summary>
    public List<string> Expand(IReadOnlyList<string> tokens)
    {
        var result = tokens.Select(t => _useDict ? DictExpansion(t) : null).ToList();

        // MWTProcessor: the seq2seq model sees every candidate, in batches of the document's order, and the dictionary
        // then overrides it (trainer.ensemble). Running only the misses would change the batches.
        if (_seq2seq != null)
        {
            var predicted = Predict(tokens);
            return result.Select((r, i) => r ?? predicted[i]).ToList();
        }

        if (_useModel)
        {
            var misses = Enumerable.Range(0, tokens.Count).Where(i => result[i] == null).ToList();
            var predicted = Predict(misses.Select(i => tokens[i]).ToList());
            for (int k = 0; k < misses.Count; k++)
                result[misses[k]] = predicted[k];
        }

        return result.Select((r, i) => r ?? tokens[i]).ToList();
    }

    /// <summary>trainer.dict_expansion: exact match, then ALL-CAPS and Capitalized forms of a lowercase entry.</summary>
    private string? DictExpansion(string word)
    {
        if (_dict.TryGetValue(word, out var expansion))
            return expansion;
        if (PyString.IsUpper(word) && _dict.TryGetValue(PyString.Lower(word), out expansion))
            return expansion.ToUpperInvariant();
        if (word.Length > 0 && PyString.IsUpper(word[..1]) && PyString.IsLower(word[1..]) && _dict.TryGetValue(PyString.Lower(word), out expansion))
            return char.ToUpperInvariant(expansion[0]) + expansion[1..];
        return null;
    }

    /// <summary>The model's expansion of each token, without the dictionary (trainer.predict).</summary>
    internal List<string> Predict(IReadOnlyList<string> tokens) => _seq2seq != null ? PredictSeq2Seq(tokens) : PredictCuts(tokens);

    /// <summary>The classifier marks characters that start a new word; a space goes before each.</summary>
    private List<string> PredictCuts(IReadOnlyList<string> tokens)
    {
        if (tokens.Count == 0)
            return [];

        // <SOS> chars <EOS>, one unit per code point; characters outside the vocabulary embed as <UNK>.
        var chars = tokens.Select(t => t.EnumerateRunes().Select(r => r.ToString()).ToArray()).ToList();
        var lengths = chars.Select(c => (long)c.Length + 2).ToArray();
        int width = (int)lengths.Max();
        var ids = new long[tokens.Count * width];
        Array.Fill(ids, _padId);
        for (int j = 0; j < chars.Count; j++)
        {
            ids[j * width] = _sosId;
            for (int k = 0; k < chars[j].Length; k++)
                ids[j * width + k + 1] = _vocab.GetValueOrDefault(chars[j][k], _unkId);
            ids[j * width + chars[j].Length + 1] = _eosId;
        }

        var logits = _net!.Forward(ids, tokens.Count, width, lengths);

        var result = new List<string>(tokens.Count);
        for (int j = 0; j < chars.Count; j++)
        {
            var sb = new StringBuilder();
            for (int k = 0; k < chars[j].Length; k++)
            {
                int at = (j * width + k + 1) * 2;
                if (logits[at + 1] > logits[at])
                    sb.Append(' ');
                sb.Append(chars[j][k]);
            }

            result.Add(sb.ToString().Trim());
        }

        return result;
    }

    /// <summary>
    /// The seq2seq model on the tokens in batches of <c>batch_size</c>, as MWTProcessor's DataLoader feeds it. Characters
    /// the vocabulary lacks get ids past it (DeltaVocab over all the tokens, in code point order), so they can be copied.
    /// An output without a space, or blank, gives the token back.
    /// </summary>
    /// <param name="onStep">For tests: called after each decoder step with its log-probs, the row width and which rows were done before it.</param>
    internal List<string> PredictSeq2Seq(IReadOnlyList<string> tokens, Action<float[], int, bool[]>? onStep = null)
    {
        var charToId = _vocab;
        var idToChar = _idToChar;
        var unknown = tokens.SelectMany(t => t.EnumerateRunes()).Where(r => !_vocab.ContainsKey(r.ToString())).Distinct().Order().Select(r => r.ToString()).ToList();
        if (unknown.Count > 0)
        {
            charToId = new Dictionary<string, int>(_vocab);
            foreach (var c in unknown)
                charToId[c] = charToId.Count;
            idToChar = [.. _idToChar, .. unknown];
        }

        var result = new List<string>(tokens.Count);
        for (int first = 0; first < tokens.Count; first += _batchSize)
        {
            var batch = tokens.Skip(first).Take(_batchSize).ToList();

            // <SOS> chars <EOS>, one unit per code point; sorted like data.sort_all: longest first, ties by later index first.
            var src = batch.Select(t => t.EnumerateRunes().Select(r => charToId[r.ToString()]).Prepend(_sosId).Append(_eosId).ToArray()).ToList();
            var order = Enumerable.Range(0, batch.Count).OrderByDescending(i => src[i].Length).ThenByDescending(i => i).ToArray();
            int width = src[order[0]].Length;
            var ids = new long[batch.Count * width];
            for (int r = 0; r < batch.Count; r++)
            {
                for (int k = 0; k < src[order[r]].Length; k++)
                    ids[r * width + k] = src[order[r]][k];
            }

            using var decoder = _seq2seq!.Encode(ids, batch.Count, width, null, order.Select(i => (long)src[i].Length).ToArray(), CancellationToken.None, BeamSize);
            var output = Seq2SeqConfig.Beam(decoder, batch.Count, BeamSize, _maxDecLen, neverDecodeUnk: true, CancellationToken.None, onStep);
            var expansions = new string[batch.Count];
            for (int r = 0; r < batch.Count; r++)
            {
                // Ids below 4 (<PAD> <UNK> <SOS> <EOS>) are dropped.
                var text = string.Concat(output[r].Where(id => id > Seq2SeqConfig.EosId).Select(id => idToChar[id]));
                var token = batch[order[r]];
                expansions[order[r]] = text.Any(c => !PyString.IsSpace(c)) && text.Contains(' ') ? text : token;
            }

            result.AddRange(expansions);
        }

        return result;
    }

    /// <summary>Document.set_mwt_expansions for one token: new words, offsets found by matching them in the token.</summary>
    private static void SetWords(Token token, string expansion)
    {
        var words = expansion.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (words.Length <= 1)
            words = [token.Text];

        token.IsMwtCandidate = false;
        token.Words.Clear();
        foreach (var w in words)
            token.Words.Add(new Word { Text = w });

        if (words.Length == 1)
        {
            (token.Words[0].StartChar, token.Words[0].EndChar) = (token.StartChar, token.EndChar);
            return;
        }

        if (token.StartChar is not int start)
            return;
        var match = Regex.Match(token.Text, "^" + string.Join(@"\s*", words.Select(w => $"({Regex.Escape(w)})")) + "$");
        if (!match.Success)
            return;
        for (int i = 0; i < words.Length; i++)
        {
            var g = match.Groups[i + 1];
            (token.Words[i].StartChar, token.Words[i].EndChar) = (start + g.Index, start + g.Index + g.Length);
        }
    }

    public void Dispose()
    {
        _net?.Dispose();
        _seq2seq?.Dispose();
    }
}
