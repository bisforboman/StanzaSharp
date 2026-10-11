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
                for (int j = 0; j < n; j++)
                    s[i * n + j] = scores[i, j];
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
            if (double.IsFinite(scores[k]) && scores[k] < min)
                min = scores[k];
        double shift = min * n;
        for (int i = 0; i < n; i++)
            scores[i * n] += shift;
        var tree = Solve(scores, n);
        if (tree.Skip(1).Count(h => h == 0) != 1)
            throw new InvalidOperationException("Rescaling by the lowest score should have prevented using multiple root edges");
        return tree;
    }

    /// <summary>
    /// chuliu_edmonds: contracts cycles one at a time (the last one Tarjan finds first), then expands. Each contracted
    /// graph is smaller than the one before, so it goes into a second n × n buffer, and the two take turns.
    /// </summary>
    internal static int[] Solve(double[] scores, int n)
    {
        var contractions = new Stack<(int[] Tree, int[] CycleLocs, int[] NoncycleLocs, int[] MetanodeHeads, int[] MetanodeDeps)>();
        var rented = ArrayPool<double>.Shared.Rent(n * n);
        var other = rented;
        try
        {
            Prepare(scores, n);
            var tree = ArgmaxRows(scores, n);
            var cycles = Tarjan(tree);
            while (cycles.Count > 0)
            {
                var cycle = cycles[^1];
                var (cycleLocs, noncycleLocs, heads, deps) = ProcessCycle(tree, cycle, scores, other);
                contractions.Push((tree, cycleLocs, noncycleLocs, heads, deps));
                (scores, other) = (other, scores);
                n = noncycleLocs.Length + 1;
                Prepare(scores, n);
                tree = ArgmaxRows(scores, n);
                cycles = Tarjan(tree);
            }
            while (contractions.Count > 0)
            {
                var (outer, cycleLocs, noncycleLocs, heads, deps) = contractions.Pop();
                tree = Expand(outer, tree, cycleLocs, noncycleLocs, heads, deps);
            }
            return tree;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(rented);
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

    private static int[] ArgmaxRows(double[] scores, int n)
    {
        var result = new int[n];
        for (int i = 0; i < n; i++)
            for (int j = 1; j < n; j++)
                if (scores[i * n + j] > scores[i * n + result[i]])
                    result[i] = j;
        return result;
    }

    /// <summary>tarjan: the cycles (strongly connected components of more than one node), in the order found.</summary>
    internal static List<bool[]> Tarjan(int[] tree)
    {
        int n = tree.Length, next = 0;
        var indices = new int[n];
        var lowlinks = new int[n];
        Array.Fill(indices, -1);
        Array.Fill(lowlinks, -1);
        var onStack = new bool[n];
        var stack = new List<int>();
        var cycles = new List<bool[]>();
        // Each node's dependents in ascending order: children[first[i]..first[i + 1]).
        var first = new int[n + 1];
        foreach (int head in tree)
            first[head + 1]++;
        for (int i = 0; i < n; i++)
            first[i + 1] += first[i];
        var children = new int[n];
        var fill = (int[])first.Clone();
        for (int d = 0; d < n; d++)
            children[fill[tree[d]]++] = d;

        // Stanza's explicit call stack: (node, whether it was entered, how many dependents were visited, the one being visited).
        var calls = new Stack<(int I, bool Entered, int Pos, int J)>();
        for (int root = 0; root < n; root++)
        {
            if (indices[root] != -1)
                continue;
            calls.Push((root, false, 0, -1));
            while (calls.Count > 0)
            {
                var (i, entered, pos, j) = calls.Pop();
                if (!entered)
                {
                    indices[i] = lowlinks[i] = next++;
                    stack.Add(i);
                    onStack[i] = true;
                }
                else
                    lowlinks[i] = Math.Min(lowlinks[i], lowlinks[j]);

                bool descended = false;
                while (pos < first[i + 1] - first[i])
                {
                    int d = children[first[i] + pos++];
                    if (indices[d] == -1)
                    {
                        calls.Push((i, true, pos, d));
                        calls.Push((d, false, 0, -1));
                        descended = true;
                        break;
                    }
                    if (onStack[d])
                        lowlinks[i] = Math.Min(lowlinks[i], indices[d]);
                }
                if (descended || lowlinks[i] != indices[i])
                    continue;
                // The root of a strongly connected component: pop it.
                var cycle = new bool[n];
                int size = 0;
                while (stack[^1] != i)
                {
                    int k = stack[^1];
                    stack.RemoveAt(stack.Count - 1);
                    onStack[k] = false;
                    cycle[k] = true;
                    size++;
                }
                stack.RemoveAt(stack.Count - 1);
                onStack[i] = false;
                cycle[i] = true;
                if (size > 0)
                    cycles.Add(cycle);
            }
        }
        return cycles;
    }

    /// <summary>process_cycle: writes the scores of the graph with the cycle contracted into one last node to <paramref name="sub"/>
    /// ((noncycle nodes + 1)² row-major).</summary>
    private static (int[] CycleLocs, int[] NoncycleLocs, int[] MetanodeHeads, int[] MetanodeDeps) ProcessCycle(
        int[] tree, bool[] cycle, double[] scores, double[] sub)
    {
        int size = tree.Length, c = cycle.Count(x => x), n = size - c;
        var cycleLocs = new int[c];
        var noncycleLocs = new int[n];
        for (int i = 0, a = 0, b = 0; i < size; i++)
            if (cycle[i])
                cycleLocs[a++] = i;
            else
                noncycleLocs[b++] = i;
        var cycleScores = new double[c];
        for (int k = 0; k < c; k++)
            cycleScores[k] = scores[cycleLocs[k] * size + tree[cycleLocs[k]]];
        double cycleScore = PairwiseSum(cycleScores, 0, c);

        // metanode_head_scores (c x n): the cycle node k taking noncycle head h instead of its cycle head.
        double HeadScore(int k, int h) => scores[cycleLocs[k] * size + noncycleLocs[h]] - cycleScores[k] + cycleScore;
        var metanodeHeads = new int[n];
        for (int h = 0; h < n; h++)
            for (int k = 1; k < c; k++)
                if (HeadScore(k, h) > HeadScore(metanodeHeads[h], h))
                    metanodeHeads[h] = k;
        // metanode_dep_scores (n x c): noncycle node d taking cycle node k as its head.
        var metanodeDeps = new int[n];
        for (int d = 0; d < n; d++)
            for (int k = 1; k < c; k++)
                if (scores[noncycleLocs[d] * size + cycleLocs[k]] > scores[noncycleLocs[d] * size + cycleLocs[metanodeDeps[d]]])
                    metanodeDeps[d] = k;

        int m = n + 1;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                sub[i * m + j] = scores[noncycleLocs[i] * size + noncycleLocs[j]];
            sub[n * m + i] = HeadScore(metanodeHeads[i], i);
            sub[i * m + n] = scores[noncycleLocs[i] * size + cycleLocs[metanodeDeps[i]]];
        }
        sub[n * m + n] = 0; // new double[,] left it 0; Prepare sets it to -inf anyway
        return (cycleLocs, noncycleLocs, metanodeHeads, metanodeDeps);
    }
    /// <summary>expand_contracted_tree: puts the solved contracted graph back around the cycle.</summary>
    private static int[] Expand(int[] tree, int[] contracted, int[] cycleLocs, int[] noncycleLocs, int[] metanodeHeads, int[] metanodeDeps)
    {
        int n = contracted.Length - 1;
        int cycleHead = contracted[n];
        var result = Enumerable.Repeat(-1, tree.Length).ToArray();
        for (int i = 0; i < n; i++)
            result[noncycleLocs[i]] = contracted[i] < n ? noncycleLocs[contracted[i]] : cycleLocs[metanodeDeps[i]];
        foreach (var i in cycleLocs)
            result[i] = tree[i];
        result[cycleLocs[metanodeHeads[cycleHead]]] = noncycleLocs[cycleHead];
        return result;
    }

    /// <summary>NumPy's float64 np.sum (pairwise summation, 8 accumulators below 128 elements).</summary>
    internal static double PairwiseSum(double[] a, int start, int n)
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
            var r = new double[8];
            Array.Copy(a, start, r, 0, 8);
            int i = 8;
            for (; i < n - n % 8; i += 8)
                for (int k = 0; k < 8; k++)
                    r[k] += a[start + i + k];
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
