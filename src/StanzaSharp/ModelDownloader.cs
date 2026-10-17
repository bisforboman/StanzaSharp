using System.Security.Cryptography;

namespace StanzaSharp;

/// <summary>
/// Downloads the Stanza models this version of StanzaSharp is verified against, from Stanza's Hugging Face repository
/// of the language (<c>stanfordnlp/stanza-en</c>, <c>stanfordnlp/stanza-de</c>), into a directory
/// <see cref="Pipeline.Load"/> can read. Loading never downloads anything; call this once.
/// </summary>
/// <example>
/// <code>
/// await ModelDownloader.DownloadAsync("models/stanza/en");                            // everything
/// await ModelDownloader.DownloadAsync("models/stanza/en", "tokenize,mwt,pos,lemma");  // only what these need
/// await ModelDownloader.DownloadAsync("models/stanza/en", new PipelineOptions { Package = "default_fast" });
/// await ModelDownloader.DownloadAsync("models/stanza/de", new PipelineOptions { Language = "de" });
/// using var nlp = Pipeline.Load("models/stanza/en");
/// </code>
/// </example>
public static class ModelDownloader
{
    /// <summary>The Stanza release whose models the golden tests verify.</summary>
    public const string StanzaVersion = "1.15.0";

    private static string BaseUrl(string language) => $"https://huggingface.co/stanfordnlp/stanza-{language}/resolve/v{StanzaVersion}/models/";

    // Every file of each language's packages in Pipeline.Packages and of Pipeline.Alternatives, and their MD5s, from Stanza 1.15.0's resources.json.
    internal static readonly Dictionary<string, (string Path, string Md5)[]> Files = new()
    {
        [Pipeline.English] =
        [
            ("tokenize/combined_nocharlm.pt", "764639a5f65cc42d84b294566ffa6215"),
            ("mwt/combined.pt", "dc569dbb26cc72b71847f9d0b8775d43"),
            ("pos/combined_charlm.pt", "a31dbf7269152bc85c3c79c5318ae848"),
            ("lemma/combined_nocharlm.pt", "8d3742b3f507e78a9e605afd72b34c92"),
            ("constituency/ptb3-revised_charlm.pt", "74dadb9e65bc7b88889550889d0a1d16"),
            ("pretrain/conll17.pt", "c339580492002fb1f759537dfbc57fb2"),
            ("forward_charlm/1billion.pt", "468b3377455fa0311565d46865f55afb"),
            ("backward_charlm/1billion.pt", "1405948b125b4264fd17509d1b6175ca"),
            ("depparse/combined_charlm.pt", "993a0eb712cefb881c684df8d9934f15"),
            ("ner/ontonotes-ww-multi_charlm.pt", "3cd5e98549640f448846389169a36812"),
            ("sentiment/sstplus_charlm.pt", "cf7e8e8f2a76c1dc236329dbc0e61cb4"),

            // default_fast's own models
            ("pos/combined_nocharlm.pt", "005fab042873317838e692c650e8e5e9"),
            ("depparse/combined_nocharlm.pt", "d155cf16050ba5fb640dc38424924346"),
            ("ner/ontonotes-ww-multi_nocharlm.pt", "e927ed0ce0960638a158c8838b1e7265"),

            // The treebank packages' (combined has default's; eslspok has no mwt and the identity lemmatizer)
            ("tokenize/ewt_nocharlm.pt", "d0145aa1ac58887bd51fc56c017278fe"),
            ("mwt/ewt.pt", "4315978c5d7002e179df725aee751ec5"),
            ("pos/ewt_charlm.pt", "d6991b157942d44dff468e2cd00bfe82"),
            ("lemma/ewt_nocharlm.pt", "6e05dfc22768b87fca86b95895d4deef"),
            ("depparse/ewt_charlm.pt", "168e503d6687ecadc1609c5f429c95ae"),
            ("tokenize/gum_nocharlm.pt", "967be4681e0f9bd464a156869cc4d3df"),
            ("mwt/gum.pt", "85857cef3a7031b5d83b3e123ded0f27"),
            ("pos/gum_charlm.pt", "b02c70afabd16c21cd46e38db455eb31"),
            ("lemma/gum_nocharlm.pt", "b15b09576a8807c2b6e3d85fa9ab0c17"),
            ("depparse/gum_charlm.pt", "6691010e4e6501da6fa6a2728e8bc3c5"),
            ("tokenize/lines_nocharlm.pt", "5f3b845efdfe7c44a70e71dcdf66d38d"),
            ("mwt/lines.pt", "45b262202b7142820e1e7b12e58c8398"),
            ("pos/lines_charlm.pt", "727f46b076c76c5d87433be90d613f21"),
            ("lemma/lines_nocharlm.pt", "7bb3141bec928e73020132f366a6a64c"),
            ("depparse/lines_charlm.pt", "b9d90427dac74a14eea084588031908b"),
            ("tokenize/partut_nocharlm.pt", "c66aa091d1be5148187de8ccbb79e1af"),
            ("mwt/partut.pt", "50aa57ff5fe536c3534488e50f541051"),
            ("pos/partut_charlm.pt", "83b413d52e521e91e58009ddfd2d47d0"),
            ("lemma/partut_nocharlm.pt", "e1f14329648a2883ed4948e2416c3684"),
            ("depparse/partut_charlm.pt", "53996edc3bcf682262a71326b2ac73b8"),
            ("tokenize/atis_nocharlm.pt", "404c52b35c297ad8b31bafd991864b51"),
            ("mwt/atis.pt", "7723a36e7ef135ada202951e7cfca0af"),
            ("pos/atis_charlm.pt", "be48972ce60043971db6f91b4fbc2e60"),
            ("lemma/atis_nocharlm.pt", "28fba07e6dd33806510eada307054ee7"),
            ("depparse/atis_charlm.pt", "1c29d854041a1ec9c31acf5e2ff4c359"),
            ("tokenize/childes_nocharlm.pt", "cd0831ca6fe7c736e99423b10721335f"),
            ("mwt/childes.pt", "0b6a21bfa6b8da3e90fbdf903e3948de"),
            ("pos/childes_charlm.pt", "3aa0b518745acbab56f5597aa4585488"),
            ("lemma/childes_nocharlm.pt", "cfce17057c0053dd595fb5bb8010b244"),
            ("depparse/childes_charlm.pt", "e6021bb09de4c6719599f2ba16b03fd5"),
            ("tokenize/eslspok_nocharlm.pt", "41daea96fefb4138103090d4ee42d236"),
            ("pos/eslspok_charlm.pt", "ca420543b5be33e904cb098cebe7de80"),
            ("depparse/eslspok_charlm.pt", "74f0bf70725a90cdf0e7b40bbdc9b1c7"),

            // The biomedical packages' (no mwt; craft and genia read biomed, mimic its own vectors and charlms)
            ("tokenize/craft_nocharlm.pt", "fc95d998b9ad6c94e4d683a2e22f4b83"),
            ("pos/craft_nocharlm.pt", "0bcb9b9712e81beb38a2ed4d7649b437"),
            ("lemma/craft_nocharlm.pt", "62758246109e9ed243f158ce9e4294c3"),
            ("depparse/craft_nocharlm.pt", "a7f21d3415745f5fe4debad7bc233eb5"),
            ("pretrain/biomed.pt", "9e038ff1a932086483da05102558a37c"),
            ("tokenize/genia_nocharlm.pt", "1f9e66486fc7e38917c817e65fa05501"),
            ("pos/genia_nocharlm.pt", "65a2bf877185e1b92747fd0a46aabe84"),
            ("lemma/genia_nocharlm.pt", "5922b24bda427f0a8578f1732964e06b"),
            ("depparse/genia_nocharlm.pt", "4c5328dae9e34f042230cf87194c1022"),
            ("tokenize/mimic_nocharlm.pt", "2ff39c4010bd9b96ecd03835206e4589"),
            ("pos/mimic_charlm.pt", "8208c89c2e4ff1abd7d06d45973091ce"),
            ("lemma/mimic_nocharlm.pt", "8a75bbe7d73e4148abf980d7d5233745"),
            ("depparse/mimic_charlm.pt", "7b254811da5f794bc60f82b73f6e7638"),
            ("pretrain/mimic.pt", "4fe1487ea48eb27af23a2a482610d0bd"),
            ("forward_charlm/mimic.pt", "3d3863e27afd67bf356354a728b0fb76"),
            ("backward_charlm/mimic.pt", "19a16ba4b8526e54ef8ab21ca33cc423"),

            // Pipeline.Alternatives (PipelineOptions.Models)
            ("constituency/ptb3_charlm.pt", "45d737c2172a5f8cb9324cc25af335f6"),
            ("pos/combined_electra-large.pt", "f7e65d440e6653ff3f5b4ae15655c154"),
            ("depparse/combined_electra-large.pt", "fffeab3811c9503ee6c0db06b3cf6b45"),
            ("ner/ontonotes-ww-multi_electra-large.pt", "5a6cd3ea92ff6cf0683de9c0914346b8"),
            ("constituency/ptb3-revised_electra-large.pt", "d17d7ebf61a9e6eff02b9db53a5dd293"),
            ("sentiment/sstplus_electra-large.pt", "1b16d4998d0e804fabbcc56ce5f29d8b"),
            ("tokenize/combined_charlm.pt", "1e2acd7f1340961385442a106c94950d"),
            ("tokenize/ewt_charlm.pt", "08e17410ba853cf519019d4ea17ec313"),
            ("tokenize/gum_charlm.pt", "5e797c73ecd132555f9769dddb694458"),
            ("tokenize/lines_charlm.pt", "8720268c9682bb80a282f560d64e0e8c"),
            ("tokenize/partut_charlm.pt", "eccecee9b79a7c02038e9cbff44602bf"),
            ("tokenize/atis_charlm.pt", "5a54394b7800d08cd0708ca11d5b9f60"),
            ("tokenize/childes_charlm.pt", "a964dbbbfe7107a6d866ff5f610e73e9"),
            ("tokenize/eslspok_charlm.pt", "41f8fa611e1aeaf0ec7fdec4b968c878"),
            ("lemma/combined_charlm.pt", "a3efc57724b629fabf427ce59d410f7e"),
            ("lemma/ewt_charlm.pt", "fb22142bdc178161966291ca017c7f34"),
            ("lemma/gum_charlm.pt", "c2c3eac4e7d2bd494cc03cfaea144a66"),
            ("lemma/lines_charlm.pt", "ef652b9b807a80467e0260684069e9ef"),
            ("lemma/partut_charlm.pt", "4d1b20ef133b0ac643cdca2f44f9dc95"),
            ("lemma/atis_charlm.pt", "836ced25ae428cb8ea19a015f64cfb2e"),
            ("lemma/childes_charlm.pt", "e7a6891511f8cd9b444a3ce31b6eb37e"),
            ("pos/ewt_nocharlm.pt", "31da311072360b086165291e99b1358a"),
            ("depparse/ewt_nocharlm.pt", "d4de2f2bfb1df4b508a66e036118aa5b"),
            ("pos/gum_nocharlm.pt", "7126837c00e6093fc28f84beb96fd92e"),
            ("depparse/gum_nocharlm.pt", "6d105ef7f3d65d00b4ff80139eab2807"),
            ("pos/lines_nocharlm.pt", "2d67edd3ed6d0ec3f9224c5ba0d95b46"),
            ("depparse/lines_nocharlm.pt", "2489849db172a3cc705fd4a902171767"),
            ("pos/partut_nocharlm.pt", "1c445ee0609cd017458a9c88b0e3359b"),
            ("depparse/partut_nocharlm.pt", "0a159f9eba80c3b15ea1a7a834d0940f"),
            ("pos/atis_nocharlm.pt", "3fe8ffa1de12518ffd09cf5e0a32a379"),
            ("depparse/atis_nocharlm.pt", "4ba5d940f287f5e6b8bdd551e8e2835b"),
            ("pos/childes_nocharlm.pt", "787a65690865ccbf29b177f6eedcee54"),
            ("depparse/childes_nocharlm.pt", "6e5ab2e8409d58bf9ff3b8d32e69c4eb"),
            ("pos/eslspok_nocharlm.pt", "1f91bc08dfc0fd69e2d7a35009be52bd"),
            ("depparse/eslspok_nocharlm.pt", "afcdcdadfbeb1fb9511a9f9a1ed2127a"),
            ("depparse/mimic_nocharlm.pt", "0fbc82facddcfcf4395a5bebf6c6ccf0"),
            ("ner/ontonotes_charlm.pt", "313fa9ec8b3b76c388c71eb456c355e7"),
            ("ner/ontonotes_nocharlm.pt", "96a6cb9857d6f71a8d6f1cd041d0edbd"),
            ("ner/combined_charlm.pt", "3f1017a2f0151b2556a105ddee17db49"),
            ("ner/i2b2.pt", "d860eeb99b2fd96e5e26fc1bb821eda5"),
            ("ner/radiology.pt", "043dd91c838e272dbde3d28f7bcc9ed2"),
            ("ner/conll03_charlm.pt", "adc994f36ee4f8fcc5e5bceb3597f02e"),
            ("ner/anatem.pt", "8b4b13670ba8cccde5645d9378071c19"),
            ("ner/bc4chemd.pt", "5cc779906b9e6fa1078b423c428b362c"),
            ("ner/bc5cdr.pt", "b0744897baf9d40165aca02f25e3f68d"),
            ("ner/bionlp13cg.pt", "980327713d51b2a161e26520be148413"),
            ("ner/jnlpba.pt", "95cb87ec78ef4f9f98697e3fc6ff6144"),
            ("ner/linnaeus.pt", "a172a0814335da3c59093d3ce67c8bea"),
            ("ner/ncbi_disease.pt", "f426680a5069aad5b7ca8145448d25da"),
            ("ner/s800.pt", "1f8d6dc302e9237c8d5db17be22de5d5"),
            ("forward_charlm/pubmed.pt", "3f734806be2c2f62c82fbb01421f78ec"),
            ("backward_charlm/pubmed.pt", "f2c42f6b2f7205d2b254d6ab0fb13303"),
            ("pretrain/glove.pt", "a44c27b595dd17c718d56b2c250f477a"),
            ("pretrain/fasttextcrawl.pt", "32e21612fdf3ea9b404ddbf663e9e438"),
        ],
        ["de"] =
        [
            ("tokenize/combined_nocharlm.pt", "a52679e759e3f6c29b749cfa8ca506be"),
            ("mwt/combined.pt", "0efb649b428cabcb89380498bb61c591"),
            ("pos/combined_charlm.pt", "757ff3b6a67c61d919a010f579cefb6c"),
            ("lemma/combined_nocharlm.pt", "207e57ac2df6fee797b163df299e0fa0"),
            ("constituency/spmrl_charlm.pt", "a46f6e39f5f34dd0a8d06c26389dbf2b"),
            ("pretrain/conll17.pt", "c489a3bcc876b2e25f4460e00401c124"),
            ("pretrain/fasttextwiki.pt", "29fe2ec4cf2d8b18c18b6aa9f5d48d78"),
            ("forward_charlm/newswiki.pt", "f2841fa3f1981a849d3ed08a723ccc22"),
            ("backward_charlm/newswiki.pt", "82f0d9a6a12c736648edbe28d87e9016"),
            ("depparse/combined_charlm.pt", "c07c3344d2635c789870b6627c6ad680"),
            ("ner/germeval2014.pt", "e2d33aecdfc079b18dbdbb06ec6d419a"),
            ("sentiment/sb10k_charlm.pt", "278b42bbd2c127d293a8197252dae1eb"),

            // default_fast's own models
            ("pos/combined_nocharlm.pt", "e2ccaf6a99711487adfe8e016592556a"),
            ("depparse/combined_nocharlm.pt", "12824bc61ad28f0377d5136745db005e"),

            // gsd's and hdt's (combined has default's)
            ("tokenize/gsd_nocharlm.pt", "9e519a21b6ca839cf0ae74d5ad6213f9"),
            ("mwt/gsd.pt", "ce401fcf73273ef02cefc757f0d08bec"),
            ("pos/gsd_charlm.pt", "b0335ee5f5ad28511020407945c2a393"),
            ("lemma/gsd_nocharlm.pt", "814edc18ce6743c2bb4acc4867140eab"),
            ("depparse/gsd_charlm.pt", "daeac687146e4dc32f3ca8f14aea56c2"),
            ("tokenize/hdt_nocharlm.pt", "106e9ec0521d8d6b1f8608b467cc32e9"),
            ("mwt/hdt.pt", "73846173795e9de1ee0546f182e0541e"),
            ("pos/hdt_charlm.pt", "5d460fe45205e40dea5b55a38097657b"),
            ("lemma/hdt_nocharlm.pt", "9fd1ffa0dafcfc9cd51b3cecb959be59"),
            ("depparse/hdt_charlm.pt", "d578c1c4614769122f2cb8481025639e"),

            // Pipeline.Alternatives (PipelineOptions.Models)
            ("tokenize/combined_charlm.pt", "84f39beb686da1053dd761caf9354949"),
            ("tokenize/gsd_charlm.pt", "339d8d827fab342e0ab92c9188da1a8b"),
            ("tokenize/hdt_charlm.pt", "4889f52a3527e252ef17afeb982de44f"),
            ("lemma/combined_charlm.pt", "72d678b7bae829693e3f643c22ae08cb"),
            ("lemma/gsd_charlm.pt", "a35012b2dd5c598080d74ddb80f39135"),
            ("lemma/hdt_charlm.pt", "512462bc2a4702c0f4e247c2e6cc032e"),
            ("pos/gsd_nocharlm.pt", "7986fcd7d260022e3d89f3e94be86dc9"),
            ("depparse/gsd_nocharlm.pt", "ca6edb23b145eb15d4cbb70aca935122"),
            ("pos/hdt_nocharlm.pt", "30eebbd85b15369b9fd435397bb0ca94"),
            ("depparse/hdt_nocharlm.pt", "bb25421e576ecc48fffbfe01b8bf235b"),
            ("ner/conll03.pt", "9ff9b34529a43b72fabd71626795952a"),
        ],
    };

    /// <summary>
    /// The files <paramref name="processors"/> (null: all) of <paramref name="package"/> of <paramref name="language"/>
    /// need: their own models, those of the processors they require (e.g. pos needs tokenize and mwt), and the shared
    /// pretrains and charlms they read. Exactly what <see cref="Pipeline.Load"/> reads for the same options, but the
    /// transformers (<see cref="Pipeline.TransformersOf"/>), which come from Hugging Face.
    /// </summary>
    internal static IEnumerable<(string Path, string Md5)> FilesFor(string? processors, string package = Pipeline.DefaultPackage, string paramName = "processors",
        string language = Pipeline.English, IReadOnlyDictionary<string, string>? models = null, bool pretagged = false)
    {
        var selected = Pipeline.SelectModels(package, processors, addRequired: true, paramName, language, models, pretagged);
        var paths = Pipeline.ModelFiles(selected, language).Select(p => p + ".pt").ToHashSet();
        var files = Files[language].Where(f => paths.Contains(f.Path)).ToList();
        if (files.Count != paths.Count)
            throw new InvalidOperationException($"No checksum for {string.Join(", ", paths.Except(files.Select(f => f.Path)))}");
        return files;
    }

    private static readonly HttpClient Http = new() { DefaultRequestHeaders = { { "User-Agent", "StanzaSharp" } } };

    /// <summary>
    /// Downloads the English <see langword="default"/> package's models (about 600 MB) into <paramref name="modelDir"/>, laid
    /// out as <c>&lt;processor&gt;/&lt;name&gt;.pt</c>. Files already there with the right checksum are kept;
    /// every new file is checked against its MD5 before it is put in place.
    /// </summary>
    /// <param name="progress">Receives one line per file, e.g. for a console.</param>
    public static Task DownloadAsync(string modelDir, IProgress<string>? progress = null, CancellationToken cancellationToken = default) =>
        DownloadAsync(modelDir, FilesFor(null), Http, progress, cancellationToken);

    /// <summary>
    /// Downloads only the models <paramref name="processors"/> need in the English <see langword="default"/> package,
    /// e.g. <c>"tokenize,mwt,pos,lemma"</c>: their own, those of the processors they require, and the shared pretrain
    /// and charlms when used. Otherwise like <see cref="DownloadAsync(string, IProgress{string}?, CancellationToken)"/>.
    /// </summary>
    /// <param name="processors">Comma-separated names from <see cref="Pipeline.AllProcessors"/>.</param>
    public static Task DownloadAsync(string modelDir, string processors, IProgress<string>? progress = null, CancellationToken cancellationToken = default) =>
        DownloadAsync(modelDir, FilesFor(processors), Http, progress, cancellationToken);

    /// <summary>
    /// Downloads exactly the models <see cref="Pipeline.Load"/> reads with the same <paramref name="options"/>: those
    /// of its <see cref="PipelineOptions.Language"/> and <see cref="PipelineOptions.Package"/> for its
    /// <see cref="PipelineOptions.Processors"/> (all of the package's when null) and the processors they require, with
    /// the models <see cref="PipelineOptions.Models"/> chooses (with <see cref="PipelineOptions.Pretagged"/>, lemma,
    /// depparse and constituency require none), and the transformer a transformer model reads (from Hugging Face, e.g.
    /// google/electra-large-discriminator's 1.3 GB, into <c>transformer/</c>). The other options are ignored. Otherwise like
    /// <see cref="DownloadAsync(string, IProgress{string}?, CancellationToken)"/>.
    /// </summary>
    /// <exception cref="ArgumentException">An unknown language, package or processor, a processor the package lacks, or an
    /// unknown or unsupported model in <see cref="PipelineOptions.Models"/>.</exception>
    /// <example>
    /// <code>
    /// var options = new PipelineOptions { Package = "default_fast" };
    /// await ModelDownloader.DownloadAsync("models/stanza/en", options);
    /// using var nlp = Pipeline.Load("models/stanza/en", options);
    /// </code>
    /// </example>
    public static async Task DownloadAsync(string modelDir, PipelineOptions options, IProgress<string>? progress = null, CancellationToken cancellationToken = default)
    {
        var files = FilesFor(options.Processors, options.Package, nameof(options), options.Language, options.Models, options.Pretagged);
        var selected = Pipeline.SelectModels(options.Package, options.Processors, addRequired: true, nameof(options), options.Language, options.Models, options.Pretagged);
        await DownloadAsync(modelDir, files, Http, progress, cancellationToken, options.Language);
        foreach (var name in Pipeline.TransformersOf(selected, options.Language))
            await DownloadTransformerAsync(modelDir, name, progress: progress, cancellationToken: cancellationToken);
    }

    internal static async Task DownloadAsync(string modelDir, IEnumerable<(string Path, string Md5)> files, HttpClient http,
        IProgress<string>? progress, CancellationToken cancellationToken, string language = Pipeline.English)
    {
        foreach (var (path, md5) in files)
            await FetchAsync(Path.Combine(modelDir, path), path, BaseUrl(language) + path, md5, Md5Async, http, progress, cancellationToken);
    }

    /// <summary>
    /// The transformers that Stanza's transformer models read, which transformers downloads from Hugging Face rather than
    /// Stanza (docs/transformer-plan.md): each repository's pinned revision and the files StanzaSharp reads, with their
    /// sha256 (Hugging Face's own checksum; these files are not in Stanza's resources.json).
    /// </summary>
    internal static readonly Dictionary<string, (string Revision, (string File, string Sha256)[] Files)> Transformers = new()
    {
        ["google/electra-large-discriminator"] = ("c13c3df7efadc2162f42588bd28eb4e187d602a5",
        [
            ("config.json", "9df6e6aeb17dbb95a44b31c3ba081dc84478f168ec0d56d1e94e397a8d876ac6"),
            ("tokenizer.json", "a23e7b62fb92e2cacc11087144ff99699f5575758cec85318a31f5295ae02f3b"),
            ("tokenizer_config.json", "a025160ef0431f1a392f6f050c1310f4c5d9fb6f275932dbccba73c4d214bf10"),
            ("pytorch_model.bin", "070894f07d07a499d73946378afa5548d64aca2418d17e67d398233692497e63"),
        ]),
    };

    /// <summary>
    /// Where a transformer's files go in a model directory: <c>transformer/&lt;repository&gt;/</c>, e.g.
    /// <c>models/stanza/en/transformer/google/electra-large-discriminator/</c>.
    /// </summary>
    internal static string TransformerDirectory(string modelDir, string name) => Path.Combine(modelDir, "transformer", Path.Combine(name.Split('/')));

    /// <summary>
    /// Downloads <paramref name="files"/> (null: all of them) of the transformer <paramref name="name"/> (a key of
    /// <see cref="Transformers"/>) at its pinned revision into <see cref="TransformerDirectory"/>, checking each against
    /// its sha256 and keeping files that already match.
    /// </summary>
    internal static async Task DownloadTransformerAsync(string modelDir, string name, IReadOnlyCollection<string>? files = null,
        IProgress<string>? progress = null, CancellationToken cancellationToken = default)
    {
        if (!Transformers.TryGetValue(name, out var transformer))
            throw new ArgumentException($"Unknown transformer '{name}'; supported: {string.Join(", ", Transformers.Keys)}", nameof(name));
        var unknown = files?.Except(transformer.Files.Select(f => f.File)).ToList();
        if (unknown is { Count: > 0 })
            throw new ArgumentException($"{name} has no file {string.Join(", ", unknown)}", nameof(files));
        var selected = transformer.Files.Where(f => files == null || files.Contains(f.File)).ToList();
        await DownloadTransformerAsync(modelDir, name, transformer.Revision, selected, Http, progress, cancellationToken);
    }

    internal static async Task DownloadTransformerAsync(string modelDir, string name, string revision, IEnumerable<(string File, string Sha256)> files,
        HttpClient http, IProgress<string>? progress, CancellationToken cancellationToken)
    {
        var directory = TransformerDirectory(modelDir, name);
        foreach (var (file, sha256) in files)
        {
            await FetchAsync(Path.Combine(directory, file), $"{name}/{file}", $"https://huggingface.co/{name}/resolve/{revision}/{file}", sha256,
                Sha256Async, http, progress, cancellationToken);
        }
    }

    // Downloads url to target unless target already has the expected checksum; a new file is checked before it is put
    // in place, so a failed or wrong download never replaces a good one.
    private static async Task FetchAsync(string target, string name, string url, string expected, Func<string, CancellationToken, Task<string>> checksum,
        HttpClient http, IProgress<string>? progress, CancellationToken cancellationToken)
    {
        if (File.Exists(target) && await checksum(target, cancellationToken) == expected)
        {
            progress?.Report($"{name}: up to date");
            return;
        }

        progress?.Report($"{name}: downloading");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        var partial = target + ".download";
        try
        {
            using (var response = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
            {
                response.EnsureSuccessStatusCode();
                await using var file = File.Create(partial);
                await response.Content.CopyToAsync(file, cancellationToken);
            }

            var actual = await checksum(partial, cancellationToken);
            if (actual != expected)
                throw new InvalidDataException($"{name}: checksum {actual} does not match the expected {expected}");
            File.Move(partial, target, overwrite: true);
        }
        finally
        {
            File.Delete(partial);
        }
    }

    /// <summary>
    /// <see cref="PipelineOptions.VerifyChecksums"/>: checks each file <see cref="Pipeline.Load"/> reads for
    /// <paramref name="models"/> of <paramref name="language"/> against its MD5. Converted models, which have no
    /// published checksum, throw.
    /// </summary>
    internal static void Verify(string modelDir, IReadOnlyDictionary<string, string> models, string language)
    {
        foreach (var path in Pipeline.ModelFiles(models, language))
        {
            var basePath = Path.Combine(modelDir, path);

            // Checkpoint.Load reads the converted files whenever the .json exists.
            if (File.Exists(basePath + ".json"))
            {
                throw new InvalidOperationException($"VerifyChecksums: {basePath}.json is a converted model, which has no published checksum. "
                    + "Load Stanza's .pt files (e.g. a ModelDownloader directory) or turn VerifyChecksums off.");
            }

            var file = basePath + ".pt"; // Load checked that it exists
            var expected = Files[language].Single(f => f.Path == path + ".pt").Md5;
            string actual;
            using (var stream = File.OpenRead(file))
                actual = Convert.ToHexStringLower(MD5.HashData(stream));
            if (actual != expected)
            {
                throw new InvalidDataException($"{file}: checksum {actual} does not match Stanza {StanzaVersion}'s {expected}. "
                    + "The file is corrupted or from another Stanza version; download it again with ModelDownloader.DownloadAsync.");
            }
        }
    }

    private static async Task<string> Md5Async(string path, CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await MD5.HashDataAsync(file, cancellationToken));
    }

    private static async Task<string> Sha256Async(string path, CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(path);
        return Convert.ToHexStringLower(await SHA256.HashDataAsync(file, cancellationToken));
    }
}
