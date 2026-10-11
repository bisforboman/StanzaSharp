- Less memory with sentiment (managed backend): its LSTM and convolutions no longer hold whole-batch buffers for the
  padded rows, so one call on 6,761 words peaks at 1.3 GB instead of 2.3 GB with all eight processors (1.1 GB instead of
  2.0 GB with `default_fast`). Outputs are unchanged, bit for bit.
