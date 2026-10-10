using System.Text.Json.Nodes;
using StanzaSharp.Nn;
using StanzaSharp.Nn.Managed;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Depparse;

/// <summary>
/// One parser batch: row b of each [size, width] id array is ROOT (id 3) then sentence b's words, 0 past its length.
/// <see cref="Texts"/> are the words after simplify_punct, without ROOT. <see cref="Charlms"/> (optional): the tagger's charlm
/// outputs (see <see cref="DependencyParser.Process"/>).
/// </summary>
internal sealed record DepparseBatch(IReadOnlyList<IReadOnlyList<string>> Texts, int Width, long[] Lengths,
    long[] Word, long[] Lemma, long[] Upos, long[] Xpos, long[] Pretrained, CharlmCache? Charlms = null);

/// <summary>
/// A batch's scores, [size, width, width] with [b, i, j] = dependent i, head j (0 is ROOT):
/// <see cref="ArcLogProbs"/> is the log-softmax over heads, padding columns included as in Stanza (defined for rows
/// i &lt; length); <see cref="Labels"/> the argmax relation (without the vocab prefix) and, if asked for,
/// <see cref="LabelScores"/> [size, width, width, relations] (both defined for i, j &lt; length).
/// </summary>
internal sealed record DepparseScores(int Width, int Relations, float[] ArcLogProbs, int[] Labels, float[]? LabelScores);

/// <summary>The parser's network, per <see cref="Backend"/>: <see cref="DepparseNet"/> or <see cref="ManagedDepparseNet"/>.</summary>
internal interface IDepparseNet : IDisposable
{
    /// <summary>GraphParser.forward_scores, then predict's log-softmax over heads and label argmax.</summary>
    /// <param name="labelScores">Also return every pair's label scores (tests).</param>
    /// <param name="ct">Checked after each charlm pass (or the character model), inside the LSTM and between the scorers' chunks.</param>
    DepparseScores Forward(DepparseBatch batch, bool labelScores, CancellationToken ct);
}

/// <summary>stanza/models/depparse/model.py <c>GraphParser</c> on TorchSharp (today's code).</summary>
internal sealed class DepparseNet : IDepparseNet
{
    private readonly Pretrain _pretrain;
    private readonly CharLanguageModel? _charlmForward, _charlmBackward;
    private readonly CharacterModel? _charModel;
    private readonly Linear? _transChar;
    private readonly Embedding _wordEmb, _lemmaEmb, _uposEmb, _xposEmb;
    private readonly Linear _transPretrained;
    private readonly HighwayLstm _lstm;
    private readonly DeepBiaffine _unlabeled, _deprel;
    private readonly DeepBiaffine? _linearizationScorer, _distanceScorer;
    private readonly Device _device = Weights.Device; // the device the model was loaded on

    public DepparseNet(Checkpoint ckpt, Pretrain pretrain, CharLanguageModel? charlmForward, CharLanguageModel? charlmBackward)
    {
        _pretrain = pretrain;
        (_charlmForward, _charlmBackward) = (charlmForward, charlmBackward);
        var config = ckpt.Root["config"]!;
        var model = ckpt.Root["model"]!;
        var vocab = ckpt.Root["vocab"]!;
        int hidden = config["hidden_dim"]!.GetValue<int>();
        int biaff = config["deep_biaff_hidden_dim"]!.GetValue<int>();
        int wordEmb = config["word_emb_dim"]!.GetValue<int>();
        int tagEmb = config["tag_emb_dim"]!.GetValue<int>();
        int transformed = config["transformed_dim"]!.GetValue<int>();
        // Stanza appends the UPOS+XPOS embedding twice where it means to add the UFeats one, so the
        // UFeats embeddings are loaded by Stanza but never used.
        int inputSize = transformed + 2 * wordEmb + 2 * tagEmb;
        if (_charlmForward != null)
            inputSize += _charlmForward.HiddenDim + _charlmBackward!.HiddenDim;
        else
        {
            _charModel = new CharacterModel(ckpt, model, config, vocab["char"]!, "charmodel.", bidirectional: false, attention: true);
            _transChar = nn.Linear(_charModel.OutputDim, transformed, hasBias: false).LoadFrom(ckpt, model, "trans_char.");
            inputSize += transformed;
        }

        int Count(string key) => Checkpoint.UnitToId(vocab[key]).Count;
        _wordEmb = nn.Embedding(Count("word"), wordEmb, padding_idx: 0).LoadFrom(ckpt, model, "word_emb.");
        _lemmaEmb = nn.Embedding(Count("lemma"), wordEmb, padding_idx: 0).LoadFrom(ckpt, model, "lemma_emb.");
        _uposEmb = nn.Embedding(Count("upos"), tagEmb, padding_idx: 0).LoadFrom(ckpt, model, "upos_emb.");
        _xposEmb = nn.Embedding(Count("xpos"), tagEmb, padding_idx: 0).LoadFrom(ckpt, model, "xpos_emb.");
        _transPretrained = nn.Linear(pretrain.Dim, transformed, hasBias: false).LoadFrom(ckpt, model, "trans_pretrained.");
        _lstm = new HighwayLstm(ckpt, model, "parserlstm", inputSize, hidden, config["num_layers"]!.GetValue<int>());

        int relations = vocab["deprel"]!["_id2unit"]!.AsArray().Count - DependencyParser.VocabPrefixSize;
        _unlabeled = new DeepBiaffine(ckpt, model, "unlabeled.", 2 * hidden, biaff, 1);
        _deprel = new DeepBiaffine(ckpt, model, "deprel.", 2 * hidden, biaff, relations);
        if (config["linearization"]!.GetValue<bool>())
            _linearizationScorer = new DeepBiaffine(ckpt, model, "linearization.", 2 * hidden, biaff, 1);
        if (config["distance"]!.GetValue<bool>())
            _distanceScorer = new DeepBiaffine(ckpt, model, "distance.", 2 * hidden, biaff, 1);
    }

    public DepparseScores Forward(DepparseBatch batch, bool labelScores, CancellationToken ct)
    {
        using var _ = torch.no_grad();
        using var scope = NewDisposeScope();
        var (unlabeled, deprel) = Scores(batch, ct);
        var labels = deprel.max(3).indexes.ToArray<long>();
        return new((int)unlabeled.shape[1], (int)deprel.shape[3], unlabeled.ToArray<float>(), labels.Select(l => (int)l).ToArray(),
            labelScores ? deprel.ToArray<float>() : null);
    }

    /// <summary>
    /// GraphParser.forward_scores, then the log-softmax over heads that predict applies:
    /// arc log-probs [batch, width, width] (dependent, head) and label scores [batch, width, width, relations].
    /// Row and column 0 are ROOT; padding columns count in the log-softmax, as in Stanza.
    /// </summary>
    private (Tensor Unlabeled, Tensor Deprel) Scores(DepparseBatch batch, CancellationToken cancellationToken)
    {
        var texts = batch.Texts;
        int size = texts.Count, width = batch.Width;
        var lengths = batch.Lengths;
        Tensor Ids(long[] ids) => torch.tensor(ids, [size, width], device: _device);

        var pos = _uposEmb.forward(Ids(batch.Upos)).add(_xposEmb.forward(Ids(batch.Xpos)), Scalars.One);
        Tensor[] chars;
        if (_charModel != null)
        {
            // ROOT is a word of the single character id ROOT_ID.
            long[] root = [CharacterModel.RootId];
            chars = [_transChar!.forward(_charModel.Forward(texts.Select(t => (IReadOnlyList<long[]>)t.Select(_charModel.CharIds).Prepend(root).ToList()).ToList()))];
        }
        else
        {
            // "\n" stands in for ROOT in the charlm input.
            var charlmText = texts.Select(t => (IReadOnlyList<string>)t.Prepend("\n").ToList()).ToList();
            var forward = Rnn.PadSequence(_charlmForward!.BuildCharRepresentation(charlmText));
            cancellationToken.ThrowIfCancellationRequested();
            chars = [forward, Rnn.PadSequence(_charlmBackward!.BuildCharRepresentation(charlmText))];
        }
        var input = cat([
            _transPretrained.forward(_pretrain.Embeddings[Ids(batch.Pretrained)]),
            _wordEmb.forward(Ids(batch.Word)),
            _lemmaEmb.forward(Ids(batch.Lemma)),
            pos,
            pos,
            .. chars,
        ], 2);
        foreach (var t in chars)
            t.Dispose();
        cancellationToken.ThrowIfCancellationRequested();
        var output = _lstm.Forward(input, lengths, disposeInput: true, cancellationToken);
        // pad_packed_sequence leaves zeros past each sentence; the scorers see them in the padding columns.
        using var widthScalar = width.ToScalar();
        var positions = arange(Scalars.Zero, widthScalar, Scalars.One, device: _device);
        var padding = positions.unsqueeze(0).ge(torch.tensor(lengths, device: _device).unsqueeze(1));
        output = output.masked_fill(padding.unsqueeze(2), Scalars.Zero);

        var unlabeled = _unlabeled.Forward(output, cancellationToken).squeeze(3);
        var deprel = _deprel.Forward(output, cancellationToken);
        var headOffset = (positions.view(1, 1, -1) - positions.view(1, -1, 1)).expand(size, -1, -1);
        if (_linearizationScorer != null)
        {
            var lin = _linearizationScorer.Forward(output, cancellationToken).squeeze(3);
            unlabeled = unlabeled.add(F.logsigmoid(lin * headOffset.sign().to_type(ScalarType.Float32)), Scalars.One);
        }
        if (_distanceScorer != null)
        {
            var dist = _distanceScorer.Forward(output, cancellationToken).squeeze(3);
            var predicted = Scalars.Softplus(dist).add(Scalars.One, Scalars.One); // 1 + softplus(dist)
            var target = headOffset.abs();
            // -log((target - predicted)^2 / 2 + 1)
            var penalty = -torch.log((target.to_type(ScalarType.Float32) - predicted).pow(Scalars.Two).div(Scalars.Two).add(Scalars.One, Scalars.One));
            unlabeled = unlabeled.add(penalty, Scalars.One);
        }
        unlabeled = unlabeled.masked_fill(eye(width, dtype: ScalarType.Bool, device: _device).unsqueeze(0), Scalars.NegativeInfinity);
        return (F.log_softmax(unlabeled, 2), deprel);
    }

    public void Dispose()
    {
        foreach (var m in new nn.Module[] { _wordEmb, _lemmaEmb, _uposEmb, _xposEmb, _transPretrained })
            m.Dispose();
        _lstm.Dispose();
        _charModel?.Dispose();
        _transChar?.Dispose();
        _unlabeled.Dispose();
        _deprel.Dispose();
        _linearizationScorer?.Dispose();
        _distanceScorer?.Dispose();
    }

    /// <summary>
    /// common/biaffine.py DeepBiaffineScorer with pairwise=True, scoring every word against every word:
    /// ReLU(W1 x) and ReLU(W2 x), each with a 1 appended, through a PairwiseBilinear.
    /// </summary>
    private sealed class DeepBiaffine : IDisposable
    {
        private const long ChunkFloats = 32 << 20; // 128 MB of intermediate per chunk
        private readonly Linear _w1, _w2;
        private readonly Tensor _weight, _bias;

        public DeepBiaffine(Checkpoint ckpt, JsonNode model, string prefix, int input, int hidden, int output)
        {
            _w1 = nn.Linear(input, hidden).LoadFrom(ckpt, model, prefix + "W1.");
            _w2 = nn.Linear(input, hidden).LoadFrom(ckpt, model, prefix + "W2.");
            _weight = ckpt.ToTensor(model[prefix + "scorer.W_bilin.weight"]); // [hidden + 1, hidden + 1, output]
            _bias = ckpt.ToTensor(model[prefix + "scorer.W_bilin.bias"]);
            if (!_weight.shape.SequenceEqual([hidden + 1, hidden + 1, output]))
                throw new InvalidOperationException($"{prefix}scorer.W_bilin.weight has shape [{string.Join(", ", _weight.shape)}]");
        }

        /// <returns>[batch, width, width, output]: [b, i, j] scores word i against word j.</returns>
        /// <remarks><paramref name="ct"/> is checked before each chunk.</remarks>
        public Tensor Forward(Tensor x, CancellationToken ct)
        {
            using var scope = NewDisposeScope();
            var input1 = AppendOne(F.relu(_w1.forward(x)));
            var input2 = AppendOne(F.relu(_w2.forward(x)));
            // The [batch, width, hidden + 1, out] intermediate is the depparse memory peak: 1.5 GB for the
            // label scorer on 250 sentences padded to 72 words, and einsum makes a permuted copy. Each
            // sentence's scores depend only on its own rows, so score a few sentences at a time.
            long batch = x.shape[0], width = x.shape[1];
            long chunk = Math.Max(1, ChunkFloats / (width * _weight.shape[1] * _weight.shape[2]));
            var output = empty([batch, width, width, _weight.shape[2]], dtype: x.dtype, device: x.device);
            for (long n = 0; n < batch; n += chunk)
            {
                ct.ThrowIfCancellationRequested();
                using var part = NewDisposeScope();
                long size = Math.Min(chunk, batch - n);
                var intermediate = einsum("NLI,IJO->NLJO", input1.narrow(0, n, size), _weight);
                output.narrow(0, n, size).copy_(einsum("NLJO,NMJ->NLMO", intermediate, input2.narrow(0, n, size)));
            }
            // In place: the label scorer's output is [batch, width, width, relations], too large to copy.
            return output.add_(_bias, Scalars.One).MoveToOuterDisposeScope();
        }

        private static Tensor AppendOne(Tensor x)
        {
            var shape = x.shape.ToArray();
            shape[^1] = 1;
            return cat([x, ones(shape, dtype: x.dtype, device: x.device)], -1);
        }

        public void Dispose()
        {
            _w1.Dispose();
            _w2.Dispose();
            _weight.Dispose();
            _bias.Dispose();
        }
    }
}
