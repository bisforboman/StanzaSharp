namespace StanzaSharp.Nn;

/// <summary>
/// <c>torch.topk(x, k, largest=True, sorted=False)</c> on one row, in the order torch's CPU kernel returns it on Linux
/// (owner's decision, 2026-10-20: coref's rough scorer depends on it). ATen's <c>topk_impl_loop</c> fills (value, index)
/// pairs and runs libstdc++'s <c>std::partial_sort</c> when <c>k * 64 &lt;= n</c>, else <c>std::nth_element</c> at
/// <c>k - 1</c>, with the comparator "x is NaN and y is not, or x &gt; y"; the first k pairs are the result. The order of
/// <c>nth_element</c>'s output (and which of several equal values make the top k) is the library's, so this ports
/// libstdc++ 13's (the torch wheels are built with GCC 13): <c>__introselect</c>, <c>__heap_select</c>,
/// <c>__insertion_sort</c> and the heap helpers of stl_algo.h and stl_heap.h, step for step.
/// </summary>
internal static class TopK
{
    /// <summary>Writes row's top <paramref name="k"/> indices (and their values) in torch's order.</summary>
    public static void Select(ReadOnlySpan<float> row, int k, Span<int> indices, Span<float> values)
    {
        if (k <= 0)
            return;
        if (k > row.Length)
            throw new ArgumentOutOfRangeException(nameof(k), $"k {k} is larger than the row's {row.Length}");
        var q = new (float Value, int Index)[row.Length];
        for (int j = 0; j < row.Length; j++)
            q[j] = (row[j], j);
        if ((long)k * 64 <= row.Length)
            PartialSort(q, k);
        else
            NthElement(q, k - 1);
        for (int j = 0; j < k; j++)
            (values[j], indices[j]) = q[j];
    }

    // ATen's comparator for largest: NaN first, then descending.
    private static bool Comp((float Value, int Index) x, (float Value, int Index) y) =>
        (float.IsNaN(x.Value) && !float.IsNaN(y.Value)) || x.Value > y.Value;

    // std::nth_element: __introselect(first, nth, last, __lg(last - first) * 2).
    private static void NthElement((float, int)[] q, int nth)
    {
        int first = 0, last = q.Length;
        if (first == last || nth == last)
            return;
        int depth = 2 * Log2(last - first);
        while (last - first > 3)
        {
            if (depth == 0)
            {
                HeapSelect(q, first, nth + 1, last);
                Swap(q, first, nth);
                return;
            }

            depth--;
            int cut = UnguardedPartitionPivot(q, first, last);
            if (cut <= nth)
                first = cut;
            else
                last = cut;
        }

        InsertionSort(q, first, last);
    }

    // std::__lg: the index of the highest set bit.
    private static int Log2(int n) => 31 - int.LeadingZeroCount(n);

    private static int UnguardedPartitionPivot((float, int)[] q, int first, int last)
    {
        int mid = first + ((last - first) / 2);
        MoveMedianToFirst(q, first, first + 1, mid, last - 1);
        return UnguardedPartition(q, first + 1, last, first);
    }

    private static void MoveMedianToFirst((float, int)[] q, int result, int a, int b, int c)
    {
        if (Comp(q[a], q[b]))
        {
            if (Comp(q[b], q[c]))
                Swap(q, result, b);
            else if (Comp(q[a], q[c]))
                Swap(q, result, c);
            else
                Swap(q, result, a);
        }
        else if (Comp(q[a], q[c]))
        {
            Swap(q, result, a);
        }
        else if (Comp(q[b], q[c]))
        {
            Swap(q, result, c);
        }
        else
        {
            Swap(q, result, b);
        }
    }

    private static int UnguardedPartition((float, int)[] q, int first, int last, int pivot)
    {
        while (true)
        {
            while (Comp(q[first], q[pivot]))
                first++;
            last--;
            while (Comp(q[pivot], q[last]))
                last--;
            if (!(first < last))
                return first;
            Swap(q, first, last);
            first++;
        }
    }

    private static void InsertionSort((float, int)[] q, int first, int last)
    {
        if (first == last)
            return;
        for (int i = first + 1; i != last; i++)
        {
            if (Comp(q[i], q[first]))
            {
                var value = q[i];
                Array.Copy(q, first, q, first + 1, i - first); // move_backward(first, i, i + 1)
                q[first] = value;
            }
            else
            {
                UnguardedLinearInsert(q, i);
            }
        }
    }

    private static void UnguardedLinearInsert((float, int)[] q, int last)
    {
        var value = q[last];
        int next = last - 1;
        while (Comp(value, q[next]))
        {
            q[last] = q[next];
            last = next;
            next--;
        }

        q[last] = value;
    }

    // std::partial_sort: __heap_select then __sort_heap over the first k.
    private static void PartialSort((float, int)[] q, int k)
    {
        HeapSelect(q, 0, k, q.Length);
        for (int last = k; last > 1;)
        {
            last--;
            PopHeap(q, 0, last, last);
        }
    }

    private static void HeapSelect((float, int)[] q, int first, int middle, int last)
    {
        MakeHeap(q, first, middle);
        for (int i = middle; i < last; i++)
        {
            if (Comp(q[i], q[first]))
                PopHeap(q, first, middle, i);
        }
    }

    private static void MakeHeap((float, int)[] q, int first, int last)
    {
        if (last - first < 2)
            return;
        int len = last - first, parent = (len - 2) / 2;
        while (true)
        {
            AdjustHeap(q, first, parent, len, q[first + parent]);
            if (parent == 0)
                return;
            parent--;
        }
    }

    private static void PopHeap((float, int)[] q, int first, int last, int result)
    {
        var value = q[result];
        q[result] = q[first];
        AdjustHeap(q, first, 0, last - first, value);
    }

    private static void AdjustHeap((float, int)[] q, int first, int hole, int len, (float, int) value)
    {
        int top = hole, second = hole;
        while (second < (len - 1) / 2)
        {
            second = 2 * (second + 1);
            if (Comp(q[first + second], q[first + second - 1]))
                second--;
            q[first + hole] = q[first + second];
            hole = second;
        }

        if ((len & 1) == 0 && second == (len - 2) / 2)
        {
            second = 2 * (second + 1);
            q[first + hole] = q[first + second - 1];
            hole = second - 1;
        }

        // __push_heap(first, hole, top, value)
        int parent = (hole - 1) / 2;
        while (hole > top && Comp(q[first + parent], value))
        {
            q[first + hole] = q[first + parent];
            hole = parent;
            parent = (hole - 1) / 2;
        }

        q[first + hole] = value;
    }

    private static void Swap((float, int)[] q, int a, int b) => (q[a], q[b]) = (q[b], q[a]);
}
