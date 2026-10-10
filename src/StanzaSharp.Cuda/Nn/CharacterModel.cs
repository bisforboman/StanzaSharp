using System.Text.Json.Nodes;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace StanzaSharp.Nn;

/// <summary>
/// A model's own character LSTM, one vector per word: what the <c>_nocharlm</c> tagger, parser and NER use
/// instead of the charlms. Port of stanza/models/common/char_model.py <c>CharacterModel</c>:
/// - with attention (tagger, parser): <c>sum_t sigmoid(attn(h_t)) * h_t</c> over the word's characters
/// - without (NER, bidirectional): the final forward and backward states, concatenated
/// Every word is its own packed sequence, so a word's vector does not depend on its batch.
/// </summary>
internal sealed class CharacterModel : IDisposable
{
    private const int UnkId = 1;

    private readonly Dictionary<string, int> _vocab;
    private readonly Embedding _charEmb;
    private readonly LSTM _lstm;
    private readonly Linear? _attn;
    private readonly Tensor _hInit, _cInit;
    private readonly int _hidden, _directions;
    private readonly Device _device = Weights.Device;

    /// <summary>The size of each word's vector: hidden × directions.</summary>
    public int OutputDim => _hidden * _directions;

    /// <inheritdoc cref="Managed.ManagedCharacterModel(Checkpoint, JsonNode, JsonNode, JsonNode, string, bool, bool)"/>
    public CharacterModel(Checkpoint ckpt, JsonNode stateDict, JsonNode config, JsonNode charVocab, string prefix, bool bidirectional, bool attention)
    {
        Managed.ManagedCharacterModel.CheckSupported(config, charVocab);
        _vocab = Checkpoint.UnitToId(charVocab);
        _hidden = config["char_hidden_dim"]!.GetValue<int>();
        _directions = bidirectional ? 2 : 1;
        int emb = config["char_emb_dim"]!.GetValue<int>();
        _charEmb = nn.Embedding(_vocab.Count, emb, padding_idx: 0).LoadFrom(ckpt, stateDict, prefix + "char_emb.");
        _lstm = nn.LSTM(emb, _hidden, batchFirst: true, bidirectional: bidirectional).LoadFrom(ckpt, stateDict, prefix + "charlstm.lstm.");
        if (attention)
            _attn = nn.Linear(OutputDim, 1, hasBias: false).LoadFrom(ckpt, stateDict, prefix + "char_attn.");
        _hInit = ckpt.ToTensor(stateDict[prefix + "charlstm_h_init"]);
        _cInit = ckpt.ToTensor(stateDict[prefix + "charlstm_c_init"]);
    }

    /// <summary>The vocab ids of a word's characters (code points, as in Python; unknown ones are UNK).</summary>
    public long[] CharIds(string word) => word.EnumerateRunes().Select(r => (long)_vocab.GetValueOrDefault(r.ToString(), UnkId)).ToArray();

    /// <summary>
    /// CharacterModel.forward over every word of a batch at once, as Stanza runs it.
    /// </summary>
    /// <param name="sentences">Each sentence's words as character ids (see <see cref="CharIds"/>).</param>
    /// <returns>[sentences, longest, <see cref="OutputDim"/>], zero past each sentence's end.</returns>
    public Tensor Forward(IReadOnlyList<IReadOnlyList<long[]>> sentences)
    {
        using var scope = NewDisposeScope();
        var perSentence = ForwardWords(sentences).split(sentences.Select(s => (long)s.Count).ToArray());
        return Rnn.PadSequence(perSentence).MoveToOuterDisposeScope();
    }

    /// <summary>Like <see cref="Forward"/>, unpadded: [words, <see cref="OutputDim"/>], the sentences' words in order.</summary>
    public Tensor ForwardWords(IReadOnlyList<IReadOnlyList<long[]>> sentences)
    {
        using var scope = NewDisposeScope();
        var words = sentences.SelectMany(s => s).ToList();
        int n = words.Count, longest = words.Max(w => w.Length);
        var ids = new long[n * longest];
        for (int i = 0; i < n; i++)
            words[i].CopyTo(ids, i * longest);
        var lengths = words.Select(w => (long)w.Length).ToArray();

        var embs = _charEmb.forward(torch.tensor(ids, [n, longest], device: _device));
        var h0 = _hInit.expand(_hInit.shape[0], n, _hidden).contiguous();
        var c0 = _cInit.expand(_cInit.shape[0], n, _hidden).contiguous();
        var output = Rnn.RunPacked(_lstm, embs, lengths, (h0, c0)); // [n, longest, hidden * directions]

        Tensor reps;
        if (_attn != null)
            reps = (output * torch.sigmoid(_attn.forward(output))).sum(1); // padding rows are zero
        else
        {
            // h[-2:]: the forward direction's last state is at each word's last character, the backward one's at its first.
            var last = torch.tensor(lengths, device: _device).sub(Scalars.One).view(n, 1, 1).expand(n, 1, OutputDim);
            reps = output.gather(1, last).squeeze(1);
            if (_directions == 2)
                reps = cat([reps.narrow(1, 0, _hidden), output.select(1, 0).narrow(1, _hidden, _hidden)], 1);
        }
        return reps.MoveToOuterDisposeScope();
    }

    public void Dispose()
    {
        _charEmb.Dispose();
        _lstm.Dispose();
        _attn?.Dispose();
        _hInit.Dispose();
        _cInit.Dispose();
    }
}
