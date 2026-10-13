- Converted `.json` checkpoints: invalid UTF-8, or an escaped lone surrogate, in a property name or string loaded
  without error and threw `InvalidOperationException` later, when a model read it; loading now throws
  `InvalidDataException`.
