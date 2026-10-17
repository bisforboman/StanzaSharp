using System.Buffers;

namespace StanzaSharp.Depparse;

/// <summary>
/// Port of stanza/models/common/chuliu_edmonds.py: the maximum spanning tree over arc scores,
/// with the same cycle order, argmax tie-breaking (first maximum) and float64 arithmetic as the
/// NumPy original, so ties and near-ties resolve the same way.
/// </summary>
internal static class ChuLiuEdmonds
{
    /// <summary>
    /// chuliu_edmonds_one_root: <paramref name="scores"/>[x, y] is the score of y heading x, row and
    /// column 0 being the root. Returns the head of every node (tree[0] = 0), with exactly one word on
    /// the root, which it gets by lowering every root arc by the smallest score times the node count.
    /// </summary>
    public static int[] OneRoot(double[,] scores)
    {
        int n = scores.GetLength(0);
        var s = ArrayPool<double>.Shared.Rent(n * n);
        try
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                    s[i * n + j] = scores[i, j];
            }

            return OneRoot(s, n);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(s);
        }
    }

    /// <summary><see cref="OneRoot(double[,])"/> on an n × n row-major matrix in <paramref name="scores"/>, which it overwrites.</summary>
    internal static int[] OneRoot(double[] scores, int n)
    {
        double min = double.PositiveInfinity;
        for (int k = 0; k < n * n; k++)
        {
            if (double.IsFinite(scores[k]) && scores[k] < min)
                min = scores[k];
        }

        double shift = min * n;
        for (int i = 0; i < n; i++)
            scores[i * n] += shift;
        var tree = Solve(scores, n);
        if (tree.Skip(1).Count(h => h == 0) != 1)
            throw new InvalidOperationException("Rescaling by the lowest score should have prevented using multiple root edges");
        return tree;
    }

    /// <summary>One contraction: the graph's tree and size before it, and process_cycle's index arrays (rented, longer than used).</summary>
    private readonly record struct Contraction(int[] Tree, int Size, int[] CycleLocs, int C, int[] NoncycleLocs, int[] MetanodeHeads, int[] MetanodeDeps)
    {
        public int N => Size - C;
    }

    /// <summary>
    /// chuliu_edmonds: contracts cycles one at a time (the last one Tarjan finds first), then expands. Each contracted
    /// graph is smaller than the one before, so it goes into a second n × n buffer, and the two take turns. Everything but
    /// the returned tree is rented.
    /// </summary>
    internal static int[] Solve(double[] scores, int n)
    {
        Stack<Contraction>? contractions = null; // most trees have no cycle
        var rented = ArrayPool<double>.Shared.Rent(n * n);
        var other = rented;
        var cycle = ArrayPool<bool>.Shared.Rent(n);
        try
        {
            var result = new int[n];
            var (tree, size) = (result, n);
            Prepare(scores, n);
            ArgmaxRows(scores, n, tree);
            while (Tarjan(tree.AsSpan(0, size), cycle.AsSpan(0, size)) > 0)
            {
                var contraction = ProcessCycle(tree, size, cycle, scores, other);
                (contractions ??= new()).Push(contraction);
                (scores, other) = (other, scores);
                size = contraction.N + 1;
                Prepare(scores, size);
                tree = ArrayPool<int>.Shared.Rent(size);
                ArgmaxRows(scores, size, tree);
            }

            while (contractions?.Count > 0)
            {
                var c = contractions.Pop();
                Expand(c, tree);
                ArrayPool<int>.Shared.Return(tree); // a contracted graph's tree, never result (the outermost)
                ArrayPool<int>.Shared.Return(c.CycleLocs);
                ArrayPool<int>.Shared.Return(c.NoncycleLocs);
                ArrayPool<int>.Shared.Return(c.MetanodeHeads);
                ArrayPool<int>.Shared.Return(c.MetanodeDeps);
                tree = c.Tree;
            }

            return result;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(rented);
            ArrayPool<bool>.Shared.Return(cycle);
        }
    }

    /// <summary>prepare_scores: no self-loops, and the root's own row allows only itself.</summary>
    private static void Prepare(double[] scores, int n)
    {
        for (int i = 0; i < n; i++)
        {
            scores[i * n + i] = double.NegativeInfinity;
            scores[i] = double.NegativeInfinity;
        }

        scores[0] = 0;
    }

    /// <summary>Each row's first maximum into <paramref name="result"/>[..n] (rented: every element is written).</summary>
    private static void ArgmaxRows(double[] scores, int n, int[] result)
    {
        for (int i = 0; i < n; i++)
        {
            result[i] = 0;
            for (int j = 1; j < n; j++)
            {
                if (scores[i * n + j] > scores[i * n + result[i]])
                    result[i] = j;
            }
        }
    }

    /// <summary>
    /// tarjan: how many cycles (strongly connected components of more than one node) <paramref name="tree"/> has. The last
    /// one found, which chuliu_edmonds contracts, is left in <paramref name="lastCycle"/> (as long as the tree).
    /// </summary>
    internal static int Tarjan(ReadOnlySpan<int> tree, Span<bool> lastCycle)
    {
        int n = tree.Length;

        // Scratch, rented (a sentence's decode called this per contraction; new arrays were ~1 KB per word on long texts):
        // indices, lowlinks, onStack (0/1), the stack, and each node's dependents in ascending order,
        // children[first[i]..first[i + 1]), with fill as the counting sort's cursor.
        var ints = ArrayPool<int>.Shared.Rent(7 * n + 1);
        var calls = ArrayPool<(int I, bool Entered, int Pos, int J)>.Shared.Rent(2 * n);
        try
        {
            var indices = ints.AsSpan(0, n);
            var lowlinks = ints.AsSpan(n, n);
            var onStack = ints.AsSpan(2 * n, n);
            var stack = ints.AsSpan(3 * n, n);
            var children = ints.AsSpan(4 * n, n);
            var fill = ints.AsSpan(5 * n, n);
            var first = ints.AsSpan(6 * n, n + 1);
            indices.Fill(-1);
            lowlinks.Fill(-1);
            onStack.Clear();
            first.Clear();
            foreach (int head in tree)
                first[head + 1]++;
            for (int i = 0; i < n; i++)
                first[i + 1] += first[i];
            first[..n].CopyTo(fill);
            for (int d = 0; d < n; d++)
                children[fill[tree[d]]++] = d;

            int cycles = 0, next = 0, depth = 0, calling = 0;

            // Stanza's explicit call stack: (node, whether it was entered, how many dependents were visited, the one being
            // visited). Each node is pushed unentered once and entered at most once per dependent left pending: under 2n.
            for (int root = 0; root < n; root++)
            {
                if (indices[root] != -1)
                    continue;
                calls[calling++] = (root, false, 0, -1);
                while (calling > 0)
                {
                    var (i, entered, pos, j) = calls[--calling];
                    if (!entered)
                    {
                        indices[i] = lowlinks[i] = next++;
                        stack[depth++] = i;
                        onStack[i] = 1;
                    }
                    else
                    {
                        lowlinks[i] = Math.Min(lowlinks[i], lowlinks[j]);
                    }

                    bool descended = false;
                    while (pos < first[i + 1] - first[i])
                    {
                        int d = children[first[i] + pos++];
                        if (indices[d] == -1)
                        {
                            calls[calling++] = (i, true, pos, d);
                            calls[calling++] = (d, false, 0, -1);
                            descended = true;
                            break;
                        }

                        if (onStack[d] != 0)
                            lowlinks[i] = Math.Min(lowlinks[i], indices[d]);
                    }

                    if (descended || lowlinks[i] != indices[i])
                        continue;

                    // The root of a strongly connected component: pop it. Only one of more than one node is a cycle.
                    int top = stack[..depth].LastIndexOf(i);
                    bool isCycle = top < depth - 1;
                    if (isCycle)
                    {
                        cycles++;
                        lastCycle.Clear();
                    }

                    for (int p = top; p < depth; p++)
                    {
                        onStack[stack[p]] = 0;
                        if (isCycle)
                            lastCycle[stack[p]] = true;
                    }

                    depth = top;
                }
            }

            return cycles;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(ints);
            ArrayPool<(int, bool, int, int)>.Shared.Return(calls);
        }
    }

    /// <summary>
    /// process_cycle: writes the scores of the graph with the cycle contracted into one last node to <paramref name="sub"/>
    /// ((noncycle nodes + 1)² row-major).
    /// </summary>
    private static Contraction ProcessCycle(int[] tree, int size, bool[] cycle, double[] scores, double[] sub)
    {
        int c = cycle.AsSpan(0, size).Count(true), n = size - c;
        var cycleLocs = ArrayPool<int>.Shared.Rent(c);
        var noncycleLocs = ArrayPool<int>.Shared.Rent(n);
        for (int i = 0, a = 0, b = 0; i < size; i++)
        {
            if (cycle[i])
                cycleLocs[a++] = i;
            else
                noncycleLocs[b++] = i;
        }

        var cycleScores = ArrayPool<double>.Shared.Rent(c);
        for (int k = 0; k < c; k++)
            cycleScores[k] = scores[cycleLocs[k] * size + tree[cycleLocs[k]]];
        double cycleScore = PairwiseSum(cycleScores, 0, c);

        // metanode_head_scores (c x n): the cycle node k taking noncycle head h instead of its cycle head.
        double HeadScore(int k, int h) => scores[cycleLocs[k] * size + noncycleLocs[h]] - cycleScores[k] + cycleScore;
        var metanodeHeads = ArrayPool<int>.Shared.Rent(n);
        for (int h = 0; h < n; h++)
        {
            metanodeHeads[h] = 0;
            for (int k = 1; k < c; k++)
            {
                if (HeadScore(k, h) > HeadScore(metanodeHeads[h], h))
                    metanodeHeads[h] = k;
            }
        }

        // metanode_dep_scores (n x c): noncycle node d taking cycle node k as its head.
        var metanodeDeps = ArrayPool<int>.Shared.Rent(n);
        for (int d = 0; d < n; d++)
        {
            metanodeDeps[d] = 0;
            for (int k = 1; k < c; k++)
            {
                if (scores[noncycleLocs[d] * size + cycleLocs[k]] > scores[noncycleLocs[d] * size + cycleLocs[metanodeDeps[d]]])
                    metanodeDeps[d] = k;
            }
        }

        int m = n + 1;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                sub[i * m + j] = scores[noncycleLocs[i] * size + noncycleLocs[j]];
            sub[n * m + i] = HeadScore(metanodeHeads[i], i);
            sub[i * m + n] = scores[noncycleLocs[i] * size + cycleLocs[metanodeDeps[i]]];
        }

        sub[n * m + n] = 0; // new double[,] left it 0; Prepare sets it to -inf anyway
        ArrayPool<double>.Shared.Return(cycleScores);
        return new(tree, size, cycleLocs, c, noncycleLocs, metanodeHeads, metanodeDeps);
    }

    /// <summary>
    /// expand_contracted_tree: puts the solved contracted graph back around the cycle, in place in the contraction's tree:
    /// its cycle nodes keep their heads but one, every other node gets its head from <paramref name="contracted"/>.
    /// </summary>
    private static void Expand(Contraction c, int[] contracted)
    {
        var (tree, n) = (c.Tree, c.N);
        int cycleHead = contracted[n];
        for (int i = 0; i < n; i++)
            tree[c.NoncycleLocs[i]] = contracted[i] < n ? c.NoncycleLocs[contracted[i]] : c.CycleLocs[c.MetanodeDeps[i]];
        tree[c.CycleLocs[c.MetanodeHeads[cycleHead]]] = c.NoncycleLocs[cycleHead];
    }

    /// <summary>NumPy's float64 np.sum (pairwise summation, 8 accumulators below 128 elements).</summary>
    internal static double PairwiseSum(ReadOnlySpan<double> a, int start, int n)
    {
        if (n < 8)
        {
            double res = 0;
            for (int i = 0; i < n; i++)
                res += a[start + i];
            return res;
        }

        if (n <= 128)
        {
            Span<double> r = stackalloc double[8];
            a.Slice(start, 8).CopyTo(r);
            int i = 8;
            for (; i < n - n % 8; i += 8)
            {
                for (int k = 0; k < 8; k++)
                    r[k] += a[start + i + k];
            }

            double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            for (; i < n; i++)
                res += a[start + i];
            return res;
        }

        int half = n / 2;
        half -= half % 8;
        return PairwiseSum(a, start, half) + PairwiseSum(a, start + half, n - half);
    }
}
