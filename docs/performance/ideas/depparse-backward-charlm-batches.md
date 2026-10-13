# Depparse's backward charlm in batches

An open idea ([list](../../performance.md#remaining-ideas)).

Today only a one-sentence document continues from the tagger's final
backward state (`CharlmCache.TryGetAlone`), because a batched row's kernel was thought to depend on its batch. Since
`Gemm.KernelBlocked` a row's step is batch-invariant (round 18 checked it bit for bit for the forward charlm), so the
parser could continue every cached sentence from the tagger's state over ROOT alone, if the cache also kept each
sentence's final cell state (4 KB per sentence). That is the whole backward pass of depparse's charlm work, not 3 steps.
Check it on the backward charlm the same way first.
