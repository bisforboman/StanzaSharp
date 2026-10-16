- `Pipeline.Load` with a model file missing from the folder threw a bare `FileNotFoundException` from whichever model
  read it first; it now names the file and the `stanzasharp download` command (or `ModelDownloader.DownloadAsync`)
  that fetches the missing files.
