# Unreleased changes

One file per user-visible change, in the folder of its CHANGELOG section: `added/`, `changed/`, `deprecated/`,
`removed/`, `fixed/` or `security/`. Name it after the change (`changes/fixed/pt-long1.md`); its text is the
CHANGELOG bullet, Markdown, exactly as it should appear:

```markdown
- `.pt` loading: an integer pickled as LONG1 that fits in 64 bits was kept as a big integer, so such a checkpoint
  failed to load.
```

Parallel pull requests add different files, so they never conflict. `python tools/changelog.py preview` shows the
next version's section; the release PR runs `python tools/changelog.py release X.Y.Z`, which writes it into
CHANGELOG.md (entries ordered by file name) and deletes the files.
