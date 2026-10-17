- Long texts allocate less garbage (managed backend): the dependency parser's Chu-Liu/Edmonds decoding works in rented
  buffers, and the constituency parser reuses its per-step score buffer and lists. Outputs are unchanged bit for bit.
