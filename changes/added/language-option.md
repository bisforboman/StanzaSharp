- `PipelineOptions.Language` (Stanza's language code, default `"en"`). German (`"de"`) models can be downloaded:
  `ModelDownloader.DownloadAsync(dir, new PipelineOptions { Language = "de" })` or `stanzasharp download --lang de`
  (into `models/stanza/de` by default), with the packages `default`, `default_fast`, `combined`, `gsd` and `hdt`.
  `Pipeline.Load` doesn't run German yet and throws `NotSupportedException`. An unknown language throws with the
  supported list, and so does `default_accurate`, whose transformer models StanzaSharp doesn't run. The
  `DownloadAsync(dir)` and `DownloadAsync(dir, processors)` overloads stay English.
