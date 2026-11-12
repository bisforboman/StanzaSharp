- `samples/StanzaSharp.Rag` embeds with all-MiniLM-L6-v2 on StanzaSharp's managed transformer encoder instead of ONNX
  Runtime, so the sample has no native library (#268). The encoder is not public yet: the sample builds only inside the
  repository.
