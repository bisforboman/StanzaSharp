- Managed backend: with `PipelineOptions.Threads` null the thread pool now uses the physical cores on machines with at
  least 4 of them (all logical processors before) and still all logical processors on smaller ones, never more than
  `Environment.ProcessorCount`. On an 8-core/16-thread desktop that cuts the slowest one-sentence calls (p90 16-36%
  lower) and gives two concurrent callers 11% more calls/s, while long texts run up to 9% slower. Output is unchanged.
