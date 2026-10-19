- `PipelineOptions.ModelPaths` and `ModelPath`: models of your own by file, Stanza's `{processor}_model_path` with
  `_pretrain_path`, `_forward_charlm_path` and `_backward_charlm_path`. A model reads the shared files of the one it
  replaces unless its `ModelPath` names others; a processor the package lacks is added. `ModelDownloader` and
  `VerifyChecksums` leave these files out.
