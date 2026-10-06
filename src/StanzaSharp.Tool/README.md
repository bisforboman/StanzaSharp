# StanzaSharp.Tool

The `stanzasharp` command downloads Stanza's English models for [StanzaSharp](https://www.nuget.org/packages/StanzaSharp),
for example in a Dockerfile or a CI job. Each file is checked against its published MD5, and files already present are
kept. It needs no native libtorch.

```
dotnet tool install -g StanzaSharp.Tool
stanzasharp download models/stanza/en
stanzasharp download models/stanza/en --package default_fast --processors tokenize,mwt,pos
```

- `DIR`: where to put the models (default `models/stanza/en`), in the layout `Pipeline.Load(DIR)` reads.
- `--package NAME`: `default` (the default) or `default_fast`.
- `--processors LIST`: only the models these processors need, e.g. `tokenize,mwt,pos,lemma`.

In a Dockerfile, `dotnet tool install --tool-path /tools StanzaSharp.Tool` avoids touching the global tool path.

Source, issues and docs: https://github.com/bisforboman/StanzaSharp
