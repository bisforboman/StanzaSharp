using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Mwt;

/// <summary>
/// Expands tokens the tokenizer marked as multi-word (<see cref="Token.IsMwtCandidate"/>) into
/// words, e.g. "don't" → "do" + "n't". Port of Stanza's MWTProcessor for a <c>force_exact_pieces</c>
/// model: a dictionary first (<c>ensemble_dict</c>), then a character classifier that predicts where
/// to cut the token (stanza/models/mwt/character_classifier.py, trainer.py).
/// </summary>
public sealed class MwtExpander : IDisposable
{
    private readonly Dictionary<string, string> _dict;
    private readonly Dictionary<string, int> _vocab;
    private readonly bool _useDict, _useModel;
    private readonly int _padId, _unkId, _sosId, _eosId;
    private readonly Embedding? _embedding;
    private readonly LSTM? _encoder;
    private readonly Linear? _hidden, _output;

    private MwtExpander(Checkpoint ckpt)
    {
        var config = ckpt.Root["config"]!;
        bool dictOnly = config["dict_only"]!.GetValue<bool>();
        _useModel = !dictOnly;
        _useDict = dictOnly || config["ensemble_dict"]?.GetValue<bool>() == true;
        _dict = ckpt.Root["dict"]!.AsObject().ToDictionary(kv => kv.Key, kv => kv.Value!.GetValue<string>());
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        (_padId, _unkId, _sosId, _eosId) = (_vocab["<PAD>"], _vocab["<UNK>"], _vocab["<SOS>"], _vocab["<EOS>"]);
        if (!_useModel)
            return;
        if (config["force_exact_pieces"]?.GetValue<bool>() != true)
            throw new NotSupportedException("MWT seq2seq models (force_exact_pieces = false) are not ported");

        var model = ckpt.Root["model"]!;
        int emb = config["emb_dim"]!.GetValue<int>();
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int layers = config["num_layers"]!.GetValue<int>();
        _embedding = nn.Embedding(config["vocab_size"]!.GetValue<int>(), emb, padding_idx: _padId).LoadFrom(ckpt, model, "embedding.");
        _encoder = nn.LSTM(emb, hidden / 2, numLayers: layers, batchFirst: true, bidirectional: true).LoadFrom(ckpt, model, "encoder.");
        _hidden = nn.Linear(hidden, hidden).LoadFrom(ckpt, model, "output_layer.0.");
        _output = nn.Linear(hidden, 2).LoadFrom(ckpt, model, "output_layer.2.");
    }

    /// <summary>Loads <c>basePath.json</c> + <c>basePath.safetensors</c>, e.g. <c>models/converted/en/mwt/combined</c>.</summary>
    public static MwtExpander Load(string basePath) => new(Checkpoint.Load(basePath));

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
        if (PyIsUpper(word) && _dict.TryGetValue(word.ToLowerInvariant(), out expansion))
            return expansion.ToUpperInvariant();
        if (word.Length > 0 && PyIsUpper(word[..1]) && PyIsLower(word[1..]) && _dict.TryGetValue(word.ToLowerInvariant(), out expansion))
            return char.ToUpperInvariant(expansion[0]) + expansion[1..];
        return null;
    }

    /// <summary>The classifier marks characters that start a new word; a space goes before each.</summary>
    internal List<string> Predict(List<string> tokens)
    {
        if (tokens.Count == 0)
            return [];
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();

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

        var src = torch.tensor(ids, [tokens.Count, width]);
        var encoded = Rnn.RunPacked(_encoder!, _embedding!.forward(src), lengths);
        var logits = _output!.forward(nn.functional.relu(_hidden!.forward(encoded)));
        var cuts = (logits[.., .., 1] > logits[.., .., 0]).data<bool>().ToArray();

        var result = new List<string>(tokens.Count);
        for (int j = 0; j < chars.Count; j++)
        {
            var sb = new StringBuilder();
            for (int k = 0; k < chars[j].Length; k++)
            {
                if (cuts[j * width + k + 1])
                    sb.Append(' ');
                sb.Append(chars[j][k]);
            }
            result.Add(sb.ToString().Trim());
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

    // Python's str.isupper()/islower(): at least one cased character, and all cased ones in that case.
    private static bool PyIsUpper(string s) => HasCase(s, UnicodeCategory.UppercaseLetter);
    private static bool PyIsLower(string s) => HasCase(s, UnicodeCategory.LowercaseLetter);

    private static bool HasCase(string s, UnicodeCategory wanted)
    {
        bool any = false;
        foreach (var c in s)
        {
            var cat = CharUnicodeInfo.GetUnicodeCategory(c);
            if (cat is UnicodeCategory.UppercaseLetter or UnicodeCategory.LowercaseLetter or UnicodeCategory.TitlecaseLetter)
            {
                if (cat != wanted)
                    return false;
                any = true;
            }
        }
        return any;
    }

    public void Dispose()
    {
        _embedding?.Dispose();
        _encoder?.Dispose();
        _hidden?.Dispose();
        _output?.Dispose();
    }
}
