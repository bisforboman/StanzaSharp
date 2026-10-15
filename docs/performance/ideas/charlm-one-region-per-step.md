# Both charlms in one region per character step

An open idea ([list](../../performance.md#remaining-ideas)), from [round 27](../round-27-whole-pipeline-profile.md).

`ManagedCharLanguageModel.BuildBoth` runs the forward model, then the backward model. Each one is a loop of
`PackedLstm.Step` calls, one parallel region per character step, about 400 steps per model on the benchmark text. The
two models step the same sequences: the same lengths and the same packed order, with the characters reversed. So step
t of both could be one `Step(works)` with two `StepRows`, each with its own `Whh`, table rows, h and c. `Step` already
takes several works (a bidirectional layer's two directions).

At 8 threads the caller spends 122 ms per call waiting for the last task of charlm regions (pos 48, ner 33, depparse
25, sentiment 16). That wait is about one task's length, once per region. With both models in a region, keep the task
the size it is today: 8 panels at 8 threads, so `TasksPerThread` × Count tasks per model, not per region. Then the
region count halves and the wait per region stays the same.

Depparse runs its charlms through `BuildCharRepresentation` and `Continue` (the backward ROOT continuation). Its pair
needs the same treatment where their lengths match.

**Exactness.** A panel's arithmetic doesn't depend on its task or on the other rows of the step (`Gemm.KernelBlocked`),
so the bits stay the same.

**Estimate.** About 60 ms at 8 threads (3% of a 1.75 s call), less at 4 threads, and nothing at 1 thread.
