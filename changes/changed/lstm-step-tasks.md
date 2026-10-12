- Managed backend: each LSTM time step is split into four tasks per thread instead of one, so a thread that runs late
  holds less of the step. On GitHub's free runners the tokenizer got up to 29% faster (3-vCPU macOS) and the whole
  pipeline 2–4% (the constituency parser up to 10%, depparse up to 6%). One-sentence `tokenize,mwt` calls take ~0.1 ms
  longer. The output is unchanged, bit for bit.
