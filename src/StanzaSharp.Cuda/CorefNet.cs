using StanzaSharp.Nn;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using F = TorchSharp.torch.nn.functional;

namespace StanzaSharp.Coref;

/// <summary>The coref head (stanza/models/coref) on TorchSharp, op for op as Stanza runs it; only the work is split in blocks.</summary>
internal sealed class CorefNet : ICorefNet
{
    private const int WordChunk = 512, RoughChunk = 256, PairChunk = 128, HeadChunk = 32;
    private static readonly Scalar LeakySlope = 0.01; // torch.nn.LeakyReLU's default

    private readonly Linear _attn, _bilinear, _hidden, _out, _span0, _span3, _span6, _zeros0, _zeros2;
    private readonly Embedding _distance, _spanEmb;
    private readonly Conv1d _conv0, _conv1;
    private readonly int _hidden_;
    private readonly Device _device = Weights.Device;

    public CorefNet(Checkpoint ckpt)
    {
        var root = ckpt.Root;
        Linear Lin(string module, string prefix)
        {
            var shape = ckpt.Shape(root[module]![prefix + "weight"]);
            return nn.Linear(shape[1], shape[0]).LoadFrom(ckpt, root[module]!, prefix);
        }

        Embedding Emb(string module, string prefix)
        {
            var shape = ckpt.Shape(root[module]![prefix + "weight"]);
            return nn.Embedding(shape[0], shape[1]).LoadFrom(ckpt, root[module]!, prefix);
        }

        Conv1d Conv(string prefix)
        {
            var shape = ckpt.Shape(root["sp"]![prefix + "weight"]);
            return nn.Conv1d(shape[1], shape[0], shape[2], padding: 1).LoadFrom(ckpt, root["sp"]!, prefix);
        }

        _attn = Lin("we", "attn.");
        _hidden_ = (int)ckpt.Shape(root["we"]!["attn.weight"])[1];
        _bilinear = Lin("rough_scorer", "bilinear.");
        _distance = Emb("pw", "distance_emb.");
        _hidden = Lin("a_scorer", "hidden.0.");
        _out = Lin("a_scorer", "out.");
        _span0 = Lin("sp", "ffnn.0.");
        _span3 = Lin("sp", "ffnn.3.");
        _span6 = Lin("sp", "ffnn.6.");
        _spanEmb = Emb("sp", "emb.");
        _conv0 = Conv("conv.0.");
        _conv1 = Conv("conv.1.");
        _zeros0 = Lin("zeros_predictor", "0.");
        _zeros2 = Lin("zeros_predictor", "2.");
    }

    private Tensor Rows(float[] values, int rows, int cols) => torch.tensor(values, [rows, cols], device: _device);

    private Tensor Leaky(Tensor x) => torch.where(x.ge(Scalars.Zero), x, x.mul(LeakySlope));

    public float[] Words(float[] subwords, int count, int[] starts, int[] ends)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        var x = Rows(subwords, count, _hidden_);
        var scores = _attn.forward(x).T; // [1, subwords]
        var words = new List<Tensor>();
        for (int first = 0; first < starts.Length; first += WordChunk)
        {
            // WordEncoder._attn_scores over the chunk's subwords: log(mask) + score, softmax, then mm.
            int last = Math.Min(starts.Length, first + WordChunk), from = starts[first], to = ends[last - 1];
            var mask = new float[(last - first) * (to - from)];
            for (int i = first; i < last; i++)
            {
                for (int t = from; t < to; t++)
                    mask[((i - first) * (to - from)) + t - from] = t >= starts[i] && t < ends[i] ? 0 : float.NegativeInfinity;
            }

            var attn = Rows(mask, last - first, to - from).add(scores.narrow(1, from, to - from), Scalars.One);
            words.Add(attn.softmax(1).mm(x.narrow(0, from, to - from)));
        }

        return cat(words, 0).ToArray<float>();
    }

    public void RoughScores(float[] words, int n, Action<int, int, float[], int> rows, CancellationToken ct)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        var mentions = Rows(words, n, _hidden_);
        var bilinear = _bilinear.forward(mentions);
        var others = mentions.T;
        for (int first = 0; first < n; first += RoughChunk)
        {
            ct.ThrowIfCancellationRequested();
            int count = Math.Min(RoughChunk, n - first);
            using var inner = NewDisposeScope();
            rows(first, count, bilinear.narrow(0, first, count).mm(others).ToArray<float>(), n);
        }
    }

    public float[] Anaphoricity(float[] words, int n, int[] top, int[] buckets, int k, CancellationToken ct)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        var mentions = Rows(words, n, _hidden_);
        var topIndices = torch.tensor(top.Select(i => (long)i).ToArray(), [n, k], device: _device);
        var distances = _distance.forward(torch.tensor(buckets.Select(i => (long)i).ToArray(), [n, k], device: _device));
        var result = new List<Tensor>();
        for (int first = 0; first < n; first += PairChunk)
        {
            ct.ThrowIfCancellationRequested();
            int count = Math.Min(PairChunk, n - first);

            // AnaphoricityScorer._get_pair_matrix and _ffnn.
            var a = mentions.narrow(0, first, count).unsqueeze(1).expand(-1, k, _hidden_);
            var b = mentions[topIndices.narrow(0, first, count)];
            var pairs = cat([a, b, a * b, distances.narrow(0, first, count)], 2);
            result.Add(_out.forward(Leaky(_hidden.forward(pairs))).squeeze(2));
        }

        return cat(result, 0).ToArray<float>();
    }

    public (float[] Start, float[] End) SpanScores(float[] words, int n, int[] heads, int[] sentStart, int[] sentLength, int width, CancellationToken ct)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        var mentions = Rows(words, n, _hidden_);
        var start = new float[heads.Length * width];
        var end = new float[heads.Length * width];
        for (int first = 0; first < heads.Length; first += HeadChunk)
        {
            ct.ThrowIfCancellationRequested();
            using var inner = NewDisposeScope();
            int count = Math.Min(HeadChunk, heads.Length - first);

            // SpanPredictor.forward: [head, word, emb(head - word)] for the words of the head's sentence, into a zero
            // [count, width, input] block, then the FFNN and both convolutions over the whole width.
            var headRows = new List<long>();
            var wordRows = new List<long>();
            var embIds = new List<long>();
            var positions = new List<long>();
            for (int r = 0; r < count; r++)
            {
                int h = first + r;
                for (int t = 0; t < sentLength[h]; t++)
                {
                    int word = sentStart[h] + t, id = heads[h] - word + 63;
                    headRows.Add(heads[h]);
                    wordRows.Add(word);
                    embIds.Add(id is < 0 or > 126 ? 127 : id);
                    positions.Add((r * width) + t);
                }
            }

            var pairMatrix = cat([mentions[torch.tensor([.. headRows], device: _device)], mentions[torch.tensor([.. wordRows], device: _device)],
                _spanEmb.forward(torch.tensor([.. embIds], device: _device))], 1);
            var padded = torch.zeros([count * width, pairMatrix.shape[1]], device: _device);
            padded.index_copy_(0, torch.tensor([.. positions], device: _device), pairMatrix);
            var x = F.relu(_span0.forward(padded.view(count, width, -1)));
            x = _span6.forward(F.relu(_span3.forward(x)));
            var res = _conv1.forward(_conv0.forward(x.permute(0, 2, 1))).permute(0, 2, 1).contiguous().ToArray<float>(); // [count, width, 2]
            for (int r = 0; r < count; r++)
            {
                for (int t = 0; t < width; t++)
                {
                    start[((first + r) * width) + t] = res[(((r * width) + t) * 2) + 0];
                    end[((first + r) * width) + t] = res[(((r * width) + t) * 2) + 1];
                }
            }
        }

        return (start, end);
    }

    public float[] Zeros(float[] words, int[] rows)
    {
        using var scope = NewDisposeScope();
        using var _ = no_grad();
        var x = Rows(words, words.Length / _hidden_, _hidden_)[torch.tensor(rows.Select(r => (long)r).ToArray(), device: _device)];
        return _zeros2.forward(F.relu(_zeros0.forward(x))).squeeze(1).ToArray<float>();
    }

    public void Dispose()
    {
        nn.Module[] modules = [_attn, _bilinear, _hidden, _out, _span0, _span3, _span6, _zeros0, _zeros2, _distance, _spanEmb, _conv0, _conv1];
        foreach (var m in modules)
            m.Dispose();
    }
}

/// <summary>Loads a <see cref="CorefResolver"/> on TorchSharp.</summary>
internal static class CorefResolverLoad
{
    extension(CorefResolver)
    {
        /// <summary>Loads e.g. <c>models/stanza/en/coref/udcoref_xlm-roberta-lora</c> on TorchSharp, reading the pipeline's transformer.</summary>
        /// <param name="device">Where the model runs; CPU by default. Load the transformer on the same device.</param>
        public static CorefResolver Load(string basePath, TransformerEmbedder transformer, Device? device = null) =>
            Weights.On(device, () => new CorefResolver(Checkpoint.Load(basePath), transformer, ckpt => new CorefNet(ckpt)));
    }
}
