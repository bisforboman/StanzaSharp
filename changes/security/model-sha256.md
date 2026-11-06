- Model downloads, `PipelineOptions.VerifyChecksums` and `LanguageIdentifierOptions.VerifyChecksums` check each Stanza
  model file against its SHA-256 at the pinned commit of Stanza's Hugging Face repository (from its LFS pointer), as
  transformer files already were, instead of the MD5 in Stanza's `resources.json`. The tool's `download --list` prints
  SHA-256s. Files already downloaded stay valid: they are the same files.
