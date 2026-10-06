using StanzaSharp;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// A forward or backward character language model used as a word feature by the POS tagger and the
/// parser. Port of stanza/models/common/char_model.py <c>CharacterLanguageModel</c> (inference only:
/// the decoder that predicts the next character is not needed).
/// </summary>
public sealed class CharLanguageModel : IDisposable
{
    // char_model.py CHARLM_START / CHARLM_END: a sentence starts with "\n", every word ends with " ".
    private const string Start = "\n", End = " ";

    private readonly Embedding _charEmb;
    private readonly LSTM _lstm;
    private readonly Tensor _hInit, _cInit;
    private readonly Dictionary<string, int> _vocab;
    private readonly int _unkId, _endId;

    public bool IsForward { get; }
    public int HiddenDim { get; }

    private CharLanguageModel(Checkpoint ckpt)
    {
        var args = ckpt.Root["args"]!;
        var state = ckpt.Root["state_dict"]!;
        if (ckpt.Root["vocab"]!["lower"]?.GetValue<bool>() == true)
            throw new NotSupportedException("Lowercasing charlm vocabularies are not ported");
        if (args["char_rec_dropout"]?.GetValue<double>() is double rec && rec != 0)
            throw new NotSupportedException("charlm with recurrent dropout (LSTMwRecDropout) is not ported");

        IsForward = ckpt.Root["is_forward_lm"]!.GetValue<bool>();
        HiddenDim = args["char_hidden_dim"]!.GetValue<int>();
        int layers = args["char_num_layers"]!.GetValue<int>();
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        _unkId = _vocab["<UNK>"];
        _endId = _vocab[End];

        var embShape = ckpt.Shape(state["char_emb.weight"]);
        _charEmb = nn.Embedding(embShape[0], embShape[1]).LoadFrom(ckpt, state, "char_emb.");
        _lstm = nn.LSTM(args["char_emb_dim"]!.GetValue<int>(), HiddenDim, numLayers: layers, batchFirst: true).LoadFrom(ckpt, state, "charlstm.lstm.");
        _hInit = ckpt.ToTensor(state["charlstm_h_init"]);
        _cInit = ckpt.ToTensor(state["charlstm_c_init"]);
    }

    /// <summary>Loads e.g. <c>models/converted/en/forward_charlm/1billion</c>.</summary>
    public static CharLanguageModel Load(string basePath) => new(Checkpoint.Load(basePath));

    /// <summary>
    /// build_char_representation: for each sentence, a [words, HiddenDim] tensor holding the LSTM state
    /// at the space after each word. A backward model reads the sentence reversed, character by character.
    /// </summary>
    public List<Tensor> BuildCharRepresentation(IReadOnlyList<IReadOnlyList<string>> sentences)
    {
        if (sentences.Count == 0)
            return [];
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();

        var ids = new List<List<long>>();
        var offsets = new List<long[]>();
        foreach (var words in sentences)
        {
            var chars = new List<long> { Id(Start) };
            var ends = new List<long>();
            var ordered = IsForward ? words : words.Reverse().Select(Reverse);
            foreach (var word in ordered)
            {
                foreach (var rune in word.EnumerateRunes())
                    chars.Add(Id(rune.ToString()));
                chars.Add(_endId);
                ends.Add(chars.Count - 1);
            }
            if (!IsForward)
                ends.Reverse();
            ids.Add(chars);
            offsets.Add(ends.ToArray());
        }

        // Packed like Stanza, so the LSTM never runs over padding. (Padding couldn't change the
        // positions we read anyway, but in a batch of mixed lengths it costs most of the time.)
        int width = ids.Max(c => c.Count);
        var flat = new long[ids.Count * width];
        Array.Fill(flat, _endId);
        for (int i = 0; i < ids.Count; i++)
            ids[i].CopyTo(flat, i * width);

        using var input = torch.tensor(flat, [ids.Count, width]);
        var h0 = _hInit.expand(_hInit.shape[0], ids.Count, HiddenDim).contiguous();
        var c0 = _cInit.expand(_cInit.shape[0], ids.Count, HiddenDim).contiguous();
        var result = Rnn.RunPackedAt(_lstm, _charEmb.forward(input), ids.Select(c => (long)c.Count).ToArray(), offsets, (h0, c0));
        scope.MoveToOuter((IEnumerable<IDisposable>)result);
        return result;
    }

    private long Id(string unit) => _vocab.TryGetValue(unit, out var id) ? id : _unkId;

    private static string Reverse(string word)
    {
        var runes = word.EnumerateRunes().ToArray();
        Array.Reverse(runes);
        return string.Concat(runes.Select(r => r.ToString()));
    }

    public void Dispose()
    {
        _charEmb.Dispose();
        _lstm.Dispose();
        _hInit.Dispose();
        _cInit.Dispose();
    }
}
