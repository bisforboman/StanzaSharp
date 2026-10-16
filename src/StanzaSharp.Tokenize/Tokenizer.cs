using System.Buffers;
using System.Text;
using System.Text.RegularExpressions;
using StanzaSharp.Nn.Managed;

namespace StanzaSharp.Tokenize;

/// <summary>
/// Splits raw text into sentences and tokens, following Stanza's TokenizeProcessor
/// (stanza/models/tokenization/data.py and utils.py).
/// </summary>
/// <remarks>
/// The model sees one unit per Unicode code point, like Python's <c>str</c>. Offsets
/// (<c>StartChar</c>/<c>EndChar</c>) are UTF-16 indices into the input string, so they differ from
/// Stanza's code-point offsets only after characters outside the BMP (e.g. emoji).
/// </remarks>
internal sealed partial class Tokenizer : IDisposable
{
    // TokenizeProcessor.MAX_SEQ_LENGTH_DEFAULT, used when the config has no max_seqlen. output_predictions windows
    // paragraphs at max(1000, max_seqlen); TokenizeProcessor.process then replaces tokens longer than max_seqlen.
    private const int MaxSeqLenDefault = 1000;
    private const string LongTokenReplacement = "<UNK>";

    // data.py: NEWLINE_WHITESPACE_RE, WHITESPACE_RE. Python's \s also matches \x1c-\x1f; .NET's doesn't.
    [GeneratedRegex(@"\n[\s\x1c-\x1f\u0080-\u009f]*\n")]
    private static partial Regex ParagraphBreak();

    // Python's \b: its word characters are str.isalnum() plus '_', so superscript digits and other
    // No/Nl characters count (3/14/2023² has no boundary after 2023) and combining marks do not,
    // unlike .NET's \w (L, Mn, Nd, Pc, and ZWJ/ZWNJ for \b). B stands for Python's \b in the patterns below.
    private const string PyWord = @"[\p{L}\p{N}_]";
    private const string B = "(?:(?<=" + PyWord + ")(?!" + PyWord + ")|(?<!" + PyWord + ")(?=" + PyWord + "))";

    // data.py: STRUCTURAL_FEATURES, in the order features are appended. Source-generated: the interpreter took ~5% of
    // the tokenizer's time (docs/performance/round-14-avx-512-kernels.md).
    [GeneratedRegex(B + @"[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*){0,2}\s*:")]
    private static partial Regex LabeledField();

    [GeneratedRegex(@"\(\d{3}\)\s?\d{3}-\d{4}|" + B + @"\d{3}-\d{3}-\d{4}" + B + "|" + B + @"x\d{3,5}" + B)]
    private static partial Regex PhoneId();

    [GeneratedRegex(B + @"\d{1,2}/\d{1,2}/\d{2,4}" + B + "|" + B
        + @"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}" + B)]
    private static partial Regex DatePattern();

    [GeneratedRegex(@"\$\s?\d[\d,]*(?:\.\d+)?")]
    private static partial Regex Currency();

    private static readonly (string Name, Regex Re)[] StructuralFeatures =
    [
        ("labeled_field", LabeledField()),
        ("phone_id", PhoneId()),
        ("date_pattern", DatePattern()),
        ("currency", Currency()),
    ];

    // utils.py: EMAIL_RAW_RE and URL_RAW_RE, combined as MASK_RE. Matches are forced into one token.
    private const string EmailRe = """"(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(?:2(?:5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"""";
    private const string UrlRe = """"(?:https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s"]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s"]{2,}|www\.[a-zA-Z0-9]+\.[^\s"]{2,})|[a-zA-Z0-9]+\.(?:gov|org|edu|net|com|co)(?:\.[^\s"]{2,})"""";

    [GeneratedRegex("(?:" + EmailRe + "|" + UrlRe + ")")]
    internal static partial Regex MaskRe();

    /// <summary>
    /// Every match of <see cref="MaskRe"/> contains one of these (the pattern is case-sensitive), so a paragraph without
    /// any is skipped. Read off the pattern's alternatives: an e-mail has a literal <c>@</c> after its local part; the
    /// four URL forms start <c>https?:\/\/</c> (<c>://</c>) or <c>www\.</c>; the last one has
    /// <c>\.(?:gov|org|edu|net|com|co)</c>, and ".com" contains ".co". <c>MaskLiterals_*</c> tests check it.
    /// </summary>
    internal static readonly SearchValues<string> MaskLiterals =
        SearchValues.Create(["@", "://", "www.", ".gov", ".org", ".edu", ".net", ".co"], StringComparison.Ordinal);

    private readonly ITokenizerNet _net;
    private readonly Dictionary<string, int> _vocab;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _vocabBySpan;
    private readonly int _unkId, _padId, _batchSize, _featDim, _maxSeqLen, _window;
    private readonly string[] _featFuncs;
    private readonly Regex?[] _featRes; // per feature: its structural pattern, or null for a per-unit one

    /// <param name="net">Builds the network from the checkpoint (a backend's, e.g. <see cref="ManagedTokenizerNet"/>).</param>
    internal Tokenizer(Checkpoint ckpt, Func<Checkpoint, ITokenizerNet> net)
    {
        _net = net(ckpt);
        var config = ckpt.Root["config"]!;
        _vocab = Checkpoint.UnitToId(ckpt.Root["vocab"]);
        _vocabBySpan = _vocab.GetAlternateLookup<ReadOnlySpan<char>>();
        _unkId = _vocab["<UNK>"];
        _padId = _vocab["<PAD>"];
        _batchSize = config["batch_size"]!.GetValue<int>();
        _featDim = config["feat_dim"]!.GetValue<int>();
        _maxSeqLen = config["max_seqlen"]?.GetValue<int>() ?? MaxSeqLenDefault;
        _window = Math.Max(MaxSeqLenDefault, _maxSeqLen);

        // data.py para_to_sentences: per-character features in config order, then the
        // position features, then the structural features in STRUCTURAL_FEATURES order.
        var funcs = config["feat_funcs"]!.AsArray().Select(f => f!.GetValue<string>()).ToHashSet();
        string[] perChar = ["space_before", "capitalized", "numeric"];
        _featFuncs =
        [
            .. config["feat_funcs"]!.AsArray().Select(f => f!.GetValue<string>()).Where(perChar.Contains),
            .. new[] { "end_of_para", "start_of_para" }.Where(funcs.Contains),
            .. StructuralFeatures.Select(s => s.Name).Where(funcs.Contains),
        ];
        var unknown = funcs.Except(_featFuncs).ToList();
        if (unknown.Count > 0)
            throw new NotSupportedException($"Unknown tokenizer feature functions: {string.Join(", ", unknown)}");
        if (_featFuncs.Length != _featDim)
            throw new InvalidOperationException($"feat_dim is {_featDim} but {_featFuncs.Length} features are configured");
        _featRes = [.. _featFuncs.Select(name => StructuralFeatures.FirstOrDefault(s => s.Name == name).Re)];
    }

    /// <summary>Loads <c>basePath.json</c> + <c>basePath.safetensors</c> (or <c>basePath.pt</c>), e.g. <c>models/converted/en/tokenize/combined_nocharlm</c>, on the managed backend.</summary>
    public static Tokenizer LoadManaged(string basePath) => new(Checkpoint.Load(basePath), ckpt => new ManagedTokenizerNet(ckpt));

    /// <param name="splitSentences">False is Stanza's <c>tokenize_no_ssplit</c>: each paragraph is one sentence.</param>
    public Document Process(string text, bool splitSentences = true, CancellationToken cancellationToken = default)
    {
        var paragraphs = SplitParagraphs(text);
        var preds = Predict(paragraphs, cancellationToken);

        var doc = new Document { Text = text };
        for (int i = 0; i < paragraphs.Count; i++)
            Decode(paragraphs[i], preds[i], doc, splitSentences);
        MarkWhitespace(doc);
        return doc;
    }

    /// <summary>
    /// TokenizeProcessor.bulk_process: tokenizes <paramref name="texts"/> joined by <c>"\n\n"</c> in one call, then
    /// gives each text its own document, with offsets relative to that text.
    /// </summary>
    public List<Document> Process(IReadOnlyList<string> texts, bool splitSentences = true, CancellationToken cancellationToken = default) =>
        Split(Process(string.Join("\n\n", texts), splitSentences, cancellationToken), texts);

    /// <summary>
    /// The second half of bulk_process: deals out <paramref name="combined"/>'s sentences (tokenized from
    /// <paramref name="texts"/> joined by <c>"\n\n"</c>) to one document per text, as Stanza does: a sentence goes to the
    /// current text while its last token ends inside it. Sentence ids continue across documents, as in Stanza.
    /// </summary>
    internal static List<Document> Split(Document combined, IReadOnlyList<string> texts)
    {
        var docs = new List<Document>(texts.Count);
        int offset = 0, next = 0;
        foreach (var text in texts)
        {
            var doc = new Document { Text = text };
            while (next < combined.Sentences.Count && combined.Sentences[next].Tokens[^1].EndChar - offset <= text.Length)
                doc.Sentences.Add(combined.Sentences[next++]);
            foreach (var token in doc.Sentences.SelectMany(s => s.Tokens))
            {
                token.StartChar -= offset;
                token.EndChar -= offset;
                foreach (var word in token.Words)
                {
                    word.StartChar -= offset;
                    word.EndChar -= offset;
                }
            }

            // The joining "\n\n" is not part of either text.
            if (doc.Sentences.Count > 0)
            {
                var last = doc.Sentences[^1].Tokens[^1];
                last.SpaceAfter = text[last.EndChar!.Value..];
                var first = doc.Sentences[0].Tokens[0];
                first.SpacesBefore = text[..first.StartChar!.Value];
            }

            docs.Add(doc);
            offset += text.Length + 2;
        }

        return docs;
    }

    /// <summary>
    /// TokenizeProcessor.process_pre_tokenized_text: one sentence per list, the tokens kept as given. The document's
    /// text is every token joined by single spaces, and the offsets point into it. Nothing is marked for MWT expansion.
    /// </summary>
    public static Document Pretokenized(IReadOnlyList<IReadOnlyList<string>> sentences)
    {
        var doc = new Document { Text = string.Join(" ", sentences.Select(s => string.Join(" ", s))) };
        int start = 0;
        foreach (var tokens in sentences)
        {
            var sent = NewSentence(doc);
            foreach (var text in tokens)
            {
                int end = start + text.Length;
                var token = new Token { Text = text, StartChar = start, EndChar = end };
                token.Words.Add(new Word { Id = sent.Tokens.Count + 1, Text = text, StartChar = start, EndChar = end });
                sent.Tokens.Add(token);
                start = end + 1;
            }

            FinishSentence(sent, doc);
        }

        MarkWhitespace(doc);
        return doc;
    }

    /// <summary>Document.mark_whitespace: the text between tokens, from the original string.</summary>
    private static void MarkWhitespace(Document doc)
    {
        var text = doc.Text!;
        var tokens = doc.Sentences.SelectMany(s => s.Tokens).ToList();
        for (int i = 0; i < tokens.Count; i++)
        {
            int end = tokens[i].EndChar!.Value;
            int next = i + 1 < tokens.Count ? tokens[i + 1].StartChar!.Value : text.Length;

            // One shared " " for the usual case, not a new string per token (24 bytes each on large documents).
            tokens[i].SpaceAfter = text.AsSpan(end, next - end) is " " ? " " : text[end..next];
        }

        if (tokens.Count > 0)
            tokens[0].SpacesBefore = text[..tokens[0].StartChar!.Value];
    }

    /// <summary>Log-probabilities [len + 1, 5] for one paragraph run alone, as Stanza pads it. For tests.</summary>
    internal float[] ParagraphLogits(string paragraph, out long[] shape)
    {
        var para = SplitParagraphs(paragraph).Single();
        shape = [para.Length + 1, Classes];
        RentInputs(para);
        try
        {
            return Run([para], [0], para.Length + 1, [para.Length + 1], default);
        }
        finally
        {
            ReturnInputs(para);
        }
    }

    // ----- input: paragraphs of code-point units -----

    /// <summary>
    /// A paragraph's units, one code point each, whitespace normalized to " " (runs of it kept once), as one string and
    /// each unit's place in it. The model's inputs are built per batch (<see cref="RentInputs"/>), not kept for the call.
    /// </summary>
    private sealed class Paragraph
    {
        public required string Text;     // the units, concatenated
        public required int[] UnitStart; // [Length + 1]: unit i is Text[UnitStart[i]..UnitStart[i + 1]]
        public required int[] Starts;    // UTF-16 offset of each unit in the original text, where it has the same length
        public bool MayHaveMask;         // contains one of MaskLiterals, so MaskRe may match

        // While its batch runs: rented, the first Length (ids) and Length * featDim (features) elements valid.
        public long[]? Ids;
        public float[]? Feats;

        public int Length => UnitStart.Length - 1;

        /// <summary>Unit i is " " (no other unit starts with a space).</summary>
        public bool IsSpace(int i) => Text[UnitStart[i]] == ' ';

        public int End(int i) => Starts[i] + UnitStart[i + 1] - UnitStart[i];

        /// <summary>
        /// The units as one char each, for the regexes Stanza runs over code points (STRUCTURAL_FEATURES, MASK_RE), so a
        /// match's index is a unit index. A unit outside the BMP becomes a BMP char of its kind for those patterns (none
        /// names a non-BMP char): ٠ (Nd: \d and a word character) for a decimal digit, ª (Lo: a word character) for
        /// another letter or number, U+FFFD otherwise. As a surrogate pair, a non-BMP letter was no word character for
        /// \b (so "𠀁C:" matched labeled_field) and counted twice for [^\s"]{2,}.
        /// </summary>
        public string CodePoints => _codePoints ??= Text.Length == Length ? Text : string.Create(Length, this, static (span, p) =>
        {
            for (int i = 0; i < span.Length; i++)
            {
                var rune = Rune.GetRuneAt(p.Text, p.UnitStart[i]);
                span[i] = rune.IsBmp ? (char)rune.Value
                    : Rune.IsDigit(rune) ? '٠'
                    : Rune.IsLetter(rune) || Rune.IsNumber(rune) ? 'ª'
                    : '�';
            }
        });

        private string? _codePoints;
    }

    private static List<Paragraph> SplitParagraphs(string text)
    {
        var result = new List<Paragraph>();
        int pos = 0;
        foreach (Match m in ParagraphBreak().Matches(text))
        {
            AddParagraph(text, pos, m.Index, result);
            pos = m.Index + m.Length;
        }

        AddParagraph(text, pos, text.Length, result);
        return result;
    }

    private static void AddParagraph(string text, int start, int end, List<Paragraph> result)
    {
        while (end > start && PyString.IsSpace(text[end - 1]))
            end--;
        if (end == start)
            return;

        // At most one unit and one char per UTF-16 char of the input: rented scratch, then exact copies.
        int max = end - start;
        var offsets = ArrayPool<int>.Shared.Rent(2 * max + 1); // unit starts in the text [0, max], then in the input
        var chars = ArrayPool<char>.Shared.Rent(max);
        try
        {
            int n = 0, length = 0, pos = start;
            foreach (var rune in text.AsSpan(start, end - start).EnumerateRunes())
            {
                bool space = rune.IsBmp && (PyString.IsSpace((char)rune.Value) || rune.Value is >= 0x80 and <= 0x9f);

                // filter_consecutive_whitespaces: keep the first of a run of spaces.
                if (!(space && length > 0 && chars[length - 1] == ' '))
                {
                    offsets[n] = length;
                    offsets[max + 1 + n] = pos;
                    n++;
                    if (space)
                        chars[length++] = ' ';
                    else
                        length += rune.EncodeToUtf16(chars.AsSpan(length));
                }

                pos += rune.Utf16SequenceLength;
            }

            offsets[n] = length;
            var units = new string(chars, 0, length);
            result.Add(new Paragraph
            {
                Text = units,
                UnitStart = offsets.AsSpan(0, n + 1).ToArray(),
                Starts = offsets.AsSpan(max + 1, n).ToArray(),
                MayHaveMask = units.AsSpan().ContainsAny(MaskLiterals),
            });
        }
        finally
        {
            ArrayPool<int>.Shared.Return(offsets);
            ArrayPool<char>.Shared.Return(chars);
        }
    }

    /// <summary>Rents and fills <paramref name="para"/>'s ids and features; <see cref="ReturnInputs"/> gives them back.</summary>
    private void RentInputs(Paragraph para)
    {
        int n = para.Length;
        var ids = para.Ids = ArrayPool<long>.Shared.Rent(n);
        var feats = para.Feats = ArrayPool<float>.Shared.Rent(n * _featDim);
        for (int i = 0; i < n; i++)
            ids[i] = _vocabBySpan.TryGetValue(para.Text.AsSpan(para.UnitStart[i], para.UnitStart[i + 1] - para.UnitStart[i]), out int id) ? id : _unkId;
        Features(para, feats.AsSpan(0, n * _featDim));
    }

    private static void ReturnInputs(Paragraph para)
    {
        ArrayPool<long>.Shared.Return(para.Ids!);
        ArrayPool<float>.Shared.Return(para.Feats!);
        (para.Ids, para.Feats) = (null, null);
    }

    private void Features(Paragraph para, Span<float> feats)
    {
        int n = para.Length;
        feats.Clear();
        for (int k = 0; k < _featDim; k++)
        {
            if (_featRes[k] is { } structural)
            {
                foreach (var m in structural.EnumerateMatches(para.CodePoints))
                {
                    for (int u = m.Index; u < m.Index + m.Length; u++)
                        feats[u * _featDim + k] = 1;
                }

                continue;
            }

            var name = _featFuncs[k];
            for (int i = 0; i < n; i++)
            {
                var rune = Rune.GetRuneAt(para.Text, para.UnitStart[i]);
                bool on = name switch
                {
                    "space_before" => para.IsSpace(i),
                    "capitalized" => PyString.IsUpper(rune),
                    "numeric" => Rune.IsDigit(rune), // NUMERIC_RE on a single code point
                    "end_of_para" => i == n - 1,
                    "start_of_para" => i == 0,
                    _ => throw new NotSupportedException(name),
                };
                if (on)
                    feats[i * _featDim + k] = 1;
            }
        }
    }

    // ----- prediction, batched exactly like utils.predict -----
    private int[][] Predict(List<Paragraph> paragraphs, CancellationToken ct)
    {
        // SortedDataset sorts by length, longest first (stable), then batches of batch_size.
        // SortedDataset.collate appends one <PAD> to each row's raw units, and the model packs each
        // row at that length (trainer.predict: lengths = len(raw)), so a row's own length + 1.
        var order = Enumerable.Range(0, paragraphs.Count).OrderByDescending(i => paragraphs[i].Length).ToArray();
        var preds = new int[paragraphs.Count][];
        for (int b = 0; b < order.Length; b += _batchSize)
        {
            ct.ThrowIfCancellationRequested();
            var rows = order[b..Math.Min(b + _batchSize, order.Length)].Select(i => paragraphs[i]).ToList();
            int maxLen = rows.Max(r => r.Length);
            int[][] rowPreds;
            foreach (var r in rows)
                RentInputs(r);
            try
            {
                if (maxLen + 1 <= _window)
                {
                    var p = Argmax(Run(rows, new int[rows.Count], maxLen + 1, rows.Select(r => r.Length + 1).ToArray(), ct), rows.Count);
                    rowPreds = rows.Select((r, j) => p[j][..r.Length]).ToArray();
                }
                else
                {
                    rowPreds = PredictWindowed(rows, ct);
                }
            }
            finally
            {
                foreach (var r in rows)
                    ReturnInputs(r);
            }

            for (int j = 0; j < rows.Count; j++)
            {
                FixLabels(rows[j], rowPreds[j]);
                preds[order[b + j]] = rowPreds[j];
            }
        }

        return preds;
    }

    /// <summary>Paragraphs over the window length run in windows that restart after the last predicted sentence end.</summary>
    private int[][] PredictWindowed(List<Paragraph> rows, CancellationToken ct)
    {
        var idx = new int[rows.Count];
        var output = rows.Select(_ => new List<int>()).ToArray();
        for (bool first = true; ; first = false)
        {
            ct.ThrowIfCancellationRequested();
            var ens = rows.Select((r, j) => Math.Min(r.Length - idx[j], _window)).ToArray();
            int width = ens.Max();

            // The first window keeps collate's raw units (own length + 1, cut to the window);
            // advance_old_batch pads every row's raw units to the batch width, so later windows
            // run every row at full width and padding reaches the backward LSTM.
            var lengths = rows.Select(r => first ? Math.Min(r.Length + 1, width) : width).ToArray();
            var run = first ? [.. Enumerable.Range(0, rows.Count)] : LiveRows(rows, idx, width);
            var runPreds = Argmax(Run([.. run.Select(j => rows[j])], [.. run.Select(j => idx[j])], width, [.. run.Select(j => lengths[j])], ct), run.Length);
            var p = new int[rows.Count][];
            for (int i = 0; i < run.Length; i++)
                p[run[i]] = runPreds[i];
            for (int j = 0; j < rows.Count; j++)
            {
                if (idx[j] >= rows[j].Length)
                    continue;
                int lastBreak = Array.FindLastIndex(p[j], x => x is 2 or 4);
                int advance = lastBreak < 0 || idx[j] >= rows[j].Length - _window ? ens[j] : lastBreak + 1;
                output[j].AddRange(p[j][..advance]);
                idx[j] += advance;
            }

            if (rows.Select((r, j) => idx[j] >= r.Length).All(x => x))
                return output.Select(o => o.ToArray()).ToArray();
        }
    }

    /// <summary>
    /// The rows a window after the first runs: those not finished (a finished row is all padding and its output is
    /// unused), plus as few finished ones as keep every live row's bits. Every row runs at the full width, so only the
    /// last row of the call can differ: Gemm.Run gives row m − 1 its single-row kernel when m % MR is 1
    /// (m = rows × width, in the LSTMs' input GEMMs and the heads). That row must use it exactly when it did with all
    /// rows: if it is the batch's last row and the batch's m % MR was 1. Finished rows go before it to make it so.
    /// </summary>
    private static int[] LiveRows(List<Paragraph> rows, int[] idx, int width)
    {
        var live = Enumerable.Range(0, rows.Count).Where(j => idx[j] < rows[j].Length).ToList();
        var finished = Enumerable.Range(0, rows.Count).Where(j => idx[j] >= rows[j].Length);
        bool singleRow = live[^1] == rows.Count - 1 && rows.Count * width % Gemm.MR == 1;
        int count = live.Count;
        while (count * width % Gemm.MR == 1 != singleRow)
            count++;
        return [.. live[..^1], .. finished.Take(count - live.Count), live[^1]];
    }

    /// <summary>Runs rows[j] from unit offsets[j], padded/truncated to width, packed at lengths[j]. Returns [rows, width, 5].</summary>
    private float[] Run(List<Paragraph> rows, int[] offsets, int width, int[] lengths, CancellationToken ct)
    {
        var ids = new long[rows.Count * width];
        Array.Fill(ids, _padId);
        var feats = new float[rows.Count * width * _featDim];
        for (int j = 0; j < rows.Count; j++)
        {
            int count = Math.Clamp(rows[j].Length - offsets[j], 0, width);
            Array.Copy(rows[j].Ids!, offsets[j], ids, j * width, count);
            Array.Copy(rows[j].Feats!, offsets[j] * _featDim, feats, j * width * _featDim, count * _featDim);
        }

        return _net.Forward(ids, feats, rows.Count, width, lengths.Select(l => (long)l).ToArray(), ct);
    }

    private const int Classes = 5;

    /// <summary>Per row and position, the class with the highest log-probability (the first on a tie, like torch.argmax).</summary>
    private static int[][] Argmax(float[] logits, int rows)
    {
        int width = logits.Length / (rows * Classes);
        var result = new int[rows][];
        for (int j = 0; j < rows; j++)
        {
            result[j] = new int[width];
            for (int i = 0; i < width; i++)
            {
                var v = logits.AsSpan((j * width + i) * Classes, Classes);
                int best = 0;
                for (int c = 1; c < Classes; c++)
                {
                    if (v[c] > v[best])
                        best = c;
                }

                result[j][i] = best;
            }
        }

        return result;
    }

    /// <summary>The paragraph always ends a sentence; e-mail addresses and URLs are kept whole.</summary>
    private static void FixLabels(Paragraph para, int[] pred)
    {
        int last = para.Length - 1;
        pred[last] = pred[last] < 2 ? 2 : pred[last] > 2 ? 4 : 2;
        if (!para.MayHaveMask)
            return;

        foreach (var m in MaskRe().EnumerateMatches(para.CodePoints))
        {
            int first = m.Index, end = m.Index + m.Length - 1;
            for (int i = first; i < end; i++)
                pred[i] = 0;
            if (pred[end] == 0)
                pred[end] = 1;
        }
    }

    // ----- output: utils.decode_predictions -----
    private void Decode(Paragraph para, int[] pred, Document doc, bool splitSentences)
    {
        Sentence? sent = null;
        int tokStart = 0;
        for (int i = 0; i < para.Length; i++)
        {
            if (pred[i] < 1)
                continue;

            // The token is units [tokStart, i] minus leading spaces; all-space tokens are dropped.
            int first = tokStart;
            while (first <= i && para.IsSpace(first))
                first++;
            int label = pred[i];
            tokStart = i + 1;
            if (first > i)
                continue;

            var text = para.Text[para.UnitStart[first]..para.UnitStart[i + 1]];

            // TokenizeProcessor.process: len(token['text']) > max_seqlen, in code points (one per unit). Only the
            // text changes; offsets and the MWT flag stay.
            if (i + 1 - first > _maxSeqLen)
                text = LongTokenReplacement;
            int start = para.Starts[first], end = para.End(i);

            sent ??= NewSentence(doc);
            var token = new Token { Text = text, StartChar = start, EndChar = end, IsMwtCandidate = label >= 3 };
            token.Words.Add(new Word { Id = sent.Tokens.Count + 1, Text = text, StartChar = start, EndChar = end });
            sent.Tokens.Add(token);

            // no_ssplit: only the paragraph's end (below) ends the sentence.
            if (label is 2 or 4 && splitSentences)
            {
                FinishSentence(sent, doc);
                sent = null;
            }
        }

        if (sent != null)
            FinishSentence(sent, doc);
    }

    private static Sentence NewSentence(Document doc)
    {
        var sent = new Sentence { SentId = doc.Sentences.Count.ToString() };
        doc.Sentences.Add(sent);
        return sent;
    }

    private static void FinishSentence(Sentence sent, Document doc) =>
        sent.Text = doc.Text![sent.Tokens[0].StartChar!.Value..sent.Tokens[^1].EndChar!.Value];

    public void Dispose() => _net.Dispose();
}
