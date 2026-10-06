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
        var s = (double[,])scores.Clone();
        double min = double.PositiveInfinity;
        foreach (var x in s)
            if (double.IsFinite(x) && x < min)
                min = x;
        double shift = min * n;
        for (int i = 0; i < n; i++)
            s[i, 0] += shift;
        var tree = Solve(s);
        if (tree.Skip(1).Count(h => h == 0) != 1)
            throw new InvalidOperationException("Rescaling by the lowest score should have prevented using multiple root edges");
        return tree;
    }

    /// <summary>chuliu_edmonds: contracts cycles one at a time (the last one Tarjan finds first), then expands.</summary>
    internal static int[] Solve(double[,] scores)
    {
        var contractions = new Stack<(int[] Tree, int[] CycleLocs, int[] NoncycleLocs, int[] MetanodeHeads, int[] MetanodeDeps)>();
        Prepare(scores);
        var tree = ArgmaxRows(scores);
        var cycles = Tarjan(tree);
        while (cycles.Count > 0)
        {
            var cycle = cycles[^1];
            var (sub, cycleLocs, noncycleLocs, heads, deps) = ProcessCycle(tree, cycle, scores);
            contractions.Push((tree, cycleLocs, noncycleLocs, heads, deps));
            scores = sub;
            Prepare(scores);
            tree = ArgmaxRows(scores);
            cycles = Tarjan(tree);
        }
        while (contractions.Count > 0)
        {
            var (outer, cycleLocs, noncycleLocs, heads, deps) = contractions.Pop();
            tree = Expand(outer, tree, cycleLocs, noncycleLocs, heads, deps);
        }
        return tree;
    }

    /// <summary>prepare_scores: no self-loops, and the root's own row allows only itself.</summary>
    private static void Prepare(double[,] scores)
    {
        int n = scores.GetLength(0);
        for (int i = 0; i < n; i++)
        {
            scores[i, i] = double.NegativeInfinity;
            scores[0, i] = double.NegativeInfinity;
        }
        scores[0, 0] = 0;
    }

    private static int[] ArgmaxRows(double[,] scores)
    {
        int n = scores.GetLength(0), m = scores.GetLength(1);
        var result = new int[n];
        for (int i = 0; i < n; i++)
            for (int j = 1; j < m; j++)
                if (scores[i, j] > scores[i, result[i]])
                    result[i] = j;
        return result;
    }

    /// <summary>tarjan: the cycles (strongly connected components of more than one node), in the order found.</summary>
    internal static List<bool[]> Tarjan(int[] tree)
    {
        int n = tree.Length, next = 0;
        var indices = Enumerable.Repeat(-1, n).ToArray();
        var lowlinks = Enumerable.Repeat(-1, n).ToArray();
        var onStack = new bool[n];
        var stack = new List<int>();
        var cycles = new List<bool[]>();

        void MaybePopCycle(int i)
        {
            if (lowlinks[i] != indices[i])
                return;
            var cycle = new bool[n];
            while (stack[^1] != i)
            {
                int j = stack[^1];
                stack.RemoveAt(stack.Count - 1);
                onStack[j] = false;
                cycle[j] = true;
            }
            stack.RemoveAt(stack.Count - 1);
            onStack[i] = false;
            cycle[i] = true;
            if (cycle.Count(c => c) > 1)
                cycles.Add(cycle);
        }

        // Stanza's explicit call stack: (node, its dependents, how many were visited, the one being visited).
        var calls = new Stack<(int I, int[]? Deps, int Pos, int J)>();
        for (int root = 0; root < n; root++)
        {
            if (indices[root] != -1)
                continue;
            calls.Push((root, null, 0, -1));
            while (calls.Count > 0)
            {
                var (i, deps, pos, j) = calls.Pop();
                if (deps == null)
                {
                    indices[i] = lowlinks[i] = next++;
                    stack.Add(i);
                    onStack[i] = true;
                    deps = Enumerable.Range(0, n).Where(d => tree[d] == i).ToArray();
                }
                else
                    lowlinks[i] = Math.Min(lowlinks[i], lowlinks[j]);

                bool descended = false;
                while (pos < deps.Length)
                {
                    int d = deps[pos++];
                    if (indices[d] == -1)
                    {
                        calls.Push((i, deps, pos, d));
                        calls.Push((d, null, 0, -1));
                        descended = true;
                        break;
                    }
                    if (onStack[d])
                        lowlinks[i] = Math.Min(lowlinks[i], indices[d]);
                }
                if (!descended)
                    MaybePopCycle(i);
            }
        }
        return cycles;
    }

    /// <summary>process_cycle: the scores of the graph with the cycle contracted into one last node.</summary>
    private static (double[,] Sub, int[] CycleLocs, int[] NoncycleLocs, int[] MetanodeHeads, int[] MetanodeDeps) ProcessCycle(
        int[] tree, bool[] cycle, double[,] scores)
    {
        var cycleLocs = Enumerable.Range(0, tree.Length).Where(i => cycle[i]).ToArray();
        var noncycleLocs = Enumerable.Range(0, tree.Length).Where(i => !cycle[i]).ToArray();
        int c = cycleLocs.Length, n = noncycleLocs.Length;
        var cycleScores = cycleLocs.Select(i => scores[i, tree[i]]).ToArray();
        double cycleScore = PairwiseSum(cycleScores, 0, c);

        // metanode_head_scores (c x n): the cycle node k taking noncycle head h instead of its cycle head.
        var headScores = new double[c, n];
        for (int k = 0; k < c; k++)
            for (int h = 0; h < n; h++)
                headScores[k, h] = scores[cycleLocs[k], noncycleLocs[h]] - cycleScores[k] + cycleScore;
        var metanodeHeads = new int[n];
        for (int h = 0; h < n; h++)
            for (int k = 1; k < c; k++)
                if (headScores[k, h] > headScores[metanodeHeads[h], h])
                    metanodeHeads[h] = k;
        // metanode_dep_scores (n x c): noncycle node d taking cycle node k as its head.
        var metanodeDeps = new int[n];
        for (int d = 0; d < n; d++)
            for (int k = 1; k < c; k++)
                if (scores[noncycleLocs[d], cycleLocs[k]] > scores[noncycleLocs[d], cycleLocs[metanodeDeps[d]]])
                    metanodeDeps[d] = k;

        var sub = new double[n + 1, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                sub[i, j] = scores[noncycleLocs[i], noncycleLocs[j]];
            sub[n, i] = headScores[metanodeHeads[i], i];
            sub[i, n] = scores[noncycleLocs[i], cycleLocs[metanodeDeps[i]]];
        }
        return (sub, cycleLocs, noncycleLocs, metanodeHeads, metanodeDeps);
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
