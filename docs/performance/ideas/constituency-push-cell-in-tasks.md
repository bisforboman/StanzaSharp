# Constituency: the stack LSTM's cell inside the GEMM's tasks

An open idea ([list](../../performance.md#remaining-ideas)), from [round 27](../round-27-whole-pipeline-profile.md).

For each layer, `ManagedConstituencyNet.StackLstm.Push` does three things:

1. it copies `[x | h]` into one row per state (serially);
2. it runs `Gemm.Run(..., rowInvariant: true)` (in parallel);
3. it loops `Act.LstmCell` over every row and every unit (serially, on the caller).

At 8 threads, steps 1 and 3 take about 64 ms of constituency's 336 ms (traced) on the caller while the workers spin.
Constituency's whole serial part (about 92 ms) is a larger cost than its join waits (48 ms). With 50 states in flight,
the regions are small and many.

A 16-column gate panel holds four complete units ([i×4, f×4, g×4, o×4], `PackedLstm.GateOrder`). So the cell for those
four units can run in the task that computed the panel, as `PackedLstm.Step` already does (an epilogue per panel and
row block). The row copies of step 1 can move into a first `ManagedThreads.For` over rows, or into the same region
before its GEMM.

**Exactness.** The cell is element-wise, so where it runs doesn't change a bit. The GEMM keeps its split and
`rowInvariant`.

**Estimate.** About 40 to 50 ms at 8 threads (13 to 15% of constituency, 2.5% of the call), less at 4 threads, and
nothing at 1 thread. At 1 thread, constituency's serial part (about 150 ms of 1.34 s) limits 8 threads to about 4.7x;
4.0 to 4.5x is measured.
