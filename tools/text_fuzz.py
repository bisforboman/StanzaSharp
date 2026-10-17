#!/usr/bin/env python3
"""
Differential text fuzzing against Python Stanza (docs/text-fuzzing.md): seeded adversarial texts go through one Stanza
pipeline (in this process) and one StanzaSharp pipeline (samples/StanzaSharp.Benchmark text-fuzz, a JSON-lines child
process), both loaded once; the CoNLL-U must be byte-identical (C#'s UTF-16 offsets converted to code points). Every
difference or crash is shrunk (ddmin over code points, keeping the first differing field) and written to the report.

  python tools/text_fuzz.py run [--lang en] [--package default] [--processors LIST] [--seed 1] [--count 200]
                                [--model P=NAME]... [--pretagged] [--models models/stanza] [--benchmark EXE]
                                [--backend managed|torch] [--threads N] [--report OUT.json] [--no-shrink]
  python tools/text_fuzz.py generate [--lang en] [--seed 1] [--count 20] [--words W,W]  # print the texts, nothing run
  python tools/text_fuzz.py texts TEXTS.json [--lang ...] [--package ...] [--no-shrink]  # a JSON list of texts

--model P=NAME (repeatable or comma-separated) replaces a processor's model (PipelineOptions.Models); Stanza then gets the
processor -> model map StanzaSharp loaded as its processors dict (package None), like `stanzasharp compare --model`.
--pretagged (PipelineOptions.Pretagged, Stanza's pretagged=True) annotates documents instead of texts: each text is tagged
by Stanza's tokenize,mwt,pos,lemma of the language's default package, some lemmas and XPOS removed (seeded by the text),
and the CoNLL-U goes to both sides (CoNLL.conll2doc + nlp.process / Conllu.Read + Process(Document)); --processors is
then e.g. depparse or lemma,depparse,constituency.

Build the C# side first: dotnet build samples/StanzaSharp.Benchmark -c Release.
"""
import argparse
import json
import random
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------------------------------- the generator

CONTROL = ["\x00", "\x01", "\x07", "\x08", "\x0b", "\x0c", "\x1b", "\x7f", "\x1c", "\x1d", "\x1e", "\x1f", "\x85",
           "\u2028", "\u2029", "\x9f", "\ufeff", "\ufffd", "\ufffe", "\uffff"]
NEWLINES = ["\n", "\r\n", "\r", "\n\n", "\r\n\r\n", "\r\r", "\n\r", "\n \n", "\n\t\n", "\n\x0b\n", "\x85", "\u2028",
            "\u2029", "\n\n\n\n"]
SPACES = [" ", "  ", "\t", "\u00a0", "\u2000", "\u2003", "\u2009", "\u200a", "\u202f", "\u205f", "\u3000", "\u1680",
          "\u180e", " \t "]
ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\u2060", "\u00ad", "\u034f", "\u180e", "\ufeff"]
EMOJI = ["\U0001F600", "\U0001F44D\U0001F3FD", "\U0001F468\u200D\U0001F469\u200D\U0001F467\u200D\U0001F466",
         "\U0001F3F3\uFE0F\u200D\U0001F308", "\U0001F1F8\U0001F1EA", "\U0001F1FA\U0001F1F8\U0001F1E9\U0001F1EA",
         "\u2764\uFE0F", "\u2764\uFE0E", "\u0031\uFE0F\u20E3", "#\uFE0F\u20E3", "\U0001F9D1\U0001F3FF\u200D\U0001F4BB",
         "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F", "\u263A", "\U0001F926\u200D\u2640\uFE0F",
         "\U0001F44B\U0001F3FB\U0001F44B\U0001F3FC", ":-)", ";)", "<3", "\u00af\\_(\u30c4)_/\u00af"]
BIDI = ["\u200e", "\u200f", "\u202a", "\u202b", "\u202c", "\u202d", "\u202e", "\u2066", "\u2067", "\u2068", "\u2069",
        "\u061c"]
RTL_WORDS = ["\u05e9\u05dc\u05d5\u05dd", "\u0645\u0631\u062d\u0628\u0627", "\u0627\u0644\u0639\u0631\u0628\u064a\u0629",
             "\u05e2\u05d1\u05e8\u05d9\u05ea", "\u0633\u0644\u0627\u0645\u060c", "\u0661\u0662\u0663"]
COMBINING = ["\u0301", "\u0300", "\u0308", "\u0327", "\u0303", "\u030a", "\u0323", "\u20dd", "\u0336", "\u0489",
             "\u0301\u0302\u0303\u0304\u0305"]
PRECOMPOSED = ["caf\u00e9", "na\u00efve", "\u00fcber", "Stra\u00dfe", "M\u00fcller", "\u00c5ngstr\u00f6m", "se\u00f1or",
               "fa\u00e7ade", "\u1e9e", "\u0130stanbul", "\u03a3\u0399\u03a3\u03a5\u03a6\u039f\u03a3", "\ufb01ne",
               "\u2126", "\u212b", "\u00bd"]
NONBMP = ["\U00010348\U00010349", "\U0001D400\U0001D401\U0001D402", "\U00020000\U00020001", "\U0001F170",
          "\U00010400\U00010428", "\U0001D7CE\U0001D7CF", "\U0001F100", "\U00013000", "\U0001D11E", "\U000E0041",
          "\U0010FFFD", "\U0001E900"]
SCRIPTS = ["\u041c\u043e\u0441\u043a\u0432\u0430", "\u6771\u4eac\u3067\u3059\u3002", "\u4f60\u597d\uff0c\u4e16\u754c",
           "\uc11c\uc6b8", "\u0e01\u0e23\u0e38\u0e07\u0e40\u0e17\u0e1e", "\u0928\u092e\u0938\u094d\u0924\u0947",
           "\u10e1\u10d0\u10e5\u10d0\u10e0\u10d7\u10d5\u10d4\u10da\u10dd", "\u0391\u03b8\u03ae\u03bd\u03b1",
           "\u0570\u0561\u0575\u0565\u0580\u0565\u0576", "\u1200\u1208", "P\u0430ypal", "\u13a0\u13a1"]
FULLWIDTH = ["\uff28\uff45\uff4c\uff4c\uff4f", "\uff11\uff12\uff13", "\uff01", "\uff1f", "\uff08\uff09", "\uff0c",
             "\u3002", "\uff0e", "\uff0d", "\uff1a"]
NO_NL = ["\u00b2", "\u00b3", "\u00b9", "\u00bc", "\u2153", "\u2160", "\u216b", "\u2070", "\u2081", "\u2460", "\u3007",
         "\u0f33", "\U00010107"]
PUNCT = ["??", "?!", "!?!", "!!!", "...", "\u2026", "....", "?!?!?!", "--", "\u2014", "\u2013", "-", "''", "``", "\"",
         "\u201c", "\u201d", "\u2018", "\u2019", "\u00ab", "\u00bb", "\u201e", "(", ")", "[", "]", "{", "}", ";", ":",
         "/", "\\", "|", "*", "&", "%", "$", "\u20ac", "\u00a3", "#", "@", "^", "~", "`", "'", ".", ",", "!", "?",
         ".?", "?.", "!.", ",,", ";;", ":)", "(!)", "[...]", "\u00bf", "\u00a1", "\u203d", "\u2e2e"]
EN_MWT = ["don't", "can't", "won't", "I'm", "you're", "it's", "we've", "they'd", "she'll", "isn't", "wouldn't've",
          "y'all", "gonna", "wanna", "cannot", "o'clock", "rock'n'roll", "Jones'", "James's", "DON'T", "Can't",
          "don\u2019t", "it\u2019s", "dont", "aint", "ain't", "shan't", "'tis", "'em", "n't", "'s", "'re"]
DE_MWT = ["im", "am", "zum", "zur", "ins", "vom", "beim", "ans", "aufs", "durchs", "f\u00fcrs", "\u00fcbers", "unterm",
          "Im", "ZUM", "hinterm", "vorm", "ums", "gibt's", "geht's", "wie's", "So'n"]
NUMBERS = ["3.14", "1,000,000", "1.000.000,50", "-42", "+1", "1e-5", "0x1F", "12:30", "12:30:45", "2026-10-16",
           "16.10.2026", "10/16/26", "1st", "2nd", "3rd", "4th", "19th-century", "$5.99", "5\u20ac", "50%", "3/4",
           "1\u00bd", "No. 7", "#1", "v2.0.1", "555-1234", "+46 70 123 45 67", "A4", "B-52", "4x4", "1990s", "'90s",
           "10km", "5'11\"", "98.6\u00b0F", "\u221242", "\u00b1"]
URLS = ["https://example.com", "http://www.example.org/path?q=1&r=2#frag", "www.example.com", "example.gov",
        "ftp://files.example.net/a.txt", "mailto:someone@example.com", "someone@example.com",
        "first.last+tag@sub.example.co.uk", "@handle", "#hashtag", "user@localhost", "a@b", "http://", "://",
        "https://example.com/very/long/path/" + "x" * 40, "http://[::1]:8080/", "https://xn--bcher-kva.example/",
        "http://example.com/\u00fcml\u00e4ut", "www.", ".gov", "example.edu.", "C:\\Users\\name\\file.txt",
        "/usr/local/bin", "file:///etc/hosts", "news.bbc.co.uk", "e.g.", "i.e.", "U.S.A.", "Ph.D.", "Mr.", "Dr.",
        "St.", "approx.", "Jan.", "z.B.", "u.a.", "bzw.", "Nr.", "S.", "ca."]
QUOTES = ["\"quoted\"", "'single'", "\u201csmart\u201d", "\u2018smart\u2019", "\u00abguillemets\u00bb",
          "\u201eGerman\u201c", "\u201aGerman\u2018", "\u300cJapanese\u300d", "''double''", "``tex''", "(parens)",
          "[brackets]", "{braces}", "\"unterminated", "'", "\"\"", "''''"]
WORD_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

FRAGMENT_SETS = {
    "control": CONTROL, "newline": NEWLINES, "space": SPACES, "zero_width": ZERO_WIDTH, "emoji": EMOJI, "bidi": BIDI,
    "rtl": RTL_WORDS, "precomposed": PRECOMPOSED, "nonbmp": NONBMP, "script": SCRIPTS, "fullwidth": FULLWIDTH,
    "no_nl": NO_NL, "punct": PUNCT, "number": NUMBERS, "url": URLS, "quote": QUOTES,
}


def base_sentences(lang):
    """Real sentences: the golden corpus and validation texts' lines (tests/golden, or tests/golden/de)."""
    folder = ROOT / "tests" / "golden" / ("de" if lang == "de" else "")
    lines = []
    for path in sorted(folder.glob("*.txt")):
        for line in path.read_bytes().decode("utf-8").splitlines():
            if 20 <= len(line.strip()) <= 400:
                lines.append(line.strip())
    return lines


def long_token(rng, length):
    kind = rng.randrange(5)
    if kind == 0:
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
    if kind == 1:
        return "".join(rng.choice(WORD_CHARS) for _ in range(length))
    if kind == 2:
        return ("https://example.com/" + "a" * length)[:length]
    if kind == 3:  # non-BMP: code points, not UTF-16 units, count
        return "".join(rng.choice(["a", "\U0001D400", "\u00e9", "e\u0301"]) for _ in range(length))
    return "-".join("ab" for _ in range(length // 3 + 1))[:length]


WORDS = []  # --words: extra fragments, about four in ten when given (seeds without it give the texts they always gave)


def fragment(rng, lang):
    if WORDS and rng.random() < 0.4:
        return rng.choice(WORDS)
    kind = rng.randrange(22)
    if kind < len(FRAGMENT_SETS):
        return rng.choice(list(FRAGMENT_SETS.values())[kind])
    if kind == 16:
        return rng.choice(DE_MWT if lang == "de" else EN_MWT)
    if kind == 17:
        return rng.choice(PRECOMPOSED) + rng.choice(COMBINING)
    if kind == 18:
        return unicodedata.normalize(rng.choice(["NFD", "NFKD", "NFKC"]), rng.choice(PRECOMPOSED))
    if kind == 19:
        return rng.choice(PUNCT) * rng.randint(2, 30)
    if kind == 20:
        return chr(rng.choice([rng.randint(0x20, 0x2FFF), rng.randint(0xA000, 0xD7FF), rng.randint(0xE000, 0xFFFF),
                               rng.randint(0x10000, 0x1FFFF), rng.randint(0x20000, 0x10FFFF)]))
    return rng.choice(["a", "I", "x"]) + rng.choice(NO_NL + COMBINING + ZERO_WIDTH)


def splice(rng, sentence, lang):
    """Fragments inserted at random positions (word boundaries mostly, inside words sometimes)."""
    chars = list(sentence)
    for _ in range(rng.randint(1, 6)):
        cut = rng.randrange(len(chars) + 1)
        if rng.random() < 0.7:  # move to a word boundary
            while cut < len(chars) and not chars[cut].isspace():
                cut += 1
        frag = fragment(rng, lang)
        if rng.random() < 0.5:
            frag = " " + frag + " "
        chars[cut:cut] = list(frag)
    return "".join(chars)


def paragraph(rng, sentences, lang):
    out = []
    for _ in range(rng.randint(1, 4)):
        s = rng.choice(sentences)
        out.append(splice(rng, s, lang) if rng.random() < 0.8 else s)
    return rng.choice([" ", " ", "  ", "\n", "\t"]).join(out)


def text(rng, sentences, lang):
    kind = rng.randrange(10)
    if kind == 0:  # tokens around max_seqlen: 100 (ewt etc.), 200 (English), 300 (partut), 400 (German); windows 1000
        length = rng.choice([100, 200, 300, 400, 1000]) + rng.randint(-2, 2)
        return rng.choice(sentences) + " " + long_token(rng, length) + " " + rng.choice(sentences)
    if kind == 1:  # a long paragraph past the 1000-unit window
        para = []
        while sum(map(len, para)) < rng.randint(1000, 4000):
            para.append(splice(rng, rng.choice(sentences), lang) if rng.random() < 0.3 else rng.choice(sentences))
        return " ".join(para)
    if kind == 2:  # many paragraphs, some empty or whitespace-only
        paras = [rng.choice(["", " ", "\t", "\u00a0", "\u3000", "x", "."]) if rng.random() < 0.3
                 else rng.choice(sentences) for _ in range(rng.randint(30, 80))]
        return "\n\n".join(paras)
    if kind == 3:  # random fragments only
        return "".join(fragment(rng, lang) + rng.choice(["", " ", " ", "\n"]) for _ in range(rng.randint(1, 40)))
    seps = NEWLINES if kind == 4 else ["\n\n"]
    out = rng.choice(["", "", " ", "\n", "\ufeff", "\n\n "])
    for i in range(rng.randint(1, 5)):
        if i:
            out += rng.choice(seps)
        out += paragraph(rng, sentences, lang)
    return out + rng.choice(["", "", "\n", "\n\n", " ", "\r\n"])


def generate(lang, seed, count):
    rng = random.Random(f"{lang}:{seed}")
    sentences = base_sentences(lang)
    return [text(rng, sentences, lang) for _ in range(count)]

# --------------------------------------------------------------------------------------------------------- running


def processors_for(lang, package):
    if package in ("default",):
        return "tokenize,mwt,pos,lemma,constituency,depparse,sentiment,ner"
    if package == "default_fast":
        return "tokenize,mwt,pos,lemma,depparse,sentiment,ner"
    return "tokenize,mwt,pos,lemma,depparse"


def resources(models, lang):
    """Stanza's resources.json in the models folder (Stanza's own download), else the one stanzasharp compare ships
    (the C# downloader writes none)."""
    path = Path(models) / "resources.json"
    return str(path if path.exists() else ROOT / "src" / "StanzaSharp.Tool" / f"stanza_resources_{lang}.json")


def stanza_pipeline(models, lang, package, processors, **kwargs):
    import stanza
    return stanza.Pipeline(lang, dir=str(models), package=package, processors=processors, download_method=None,
                           resources_filepath=resources(models, lang), use_gpu=False, logging_level="WARN", **kwargs)


class Stanza:
    def __init__(self, models, lang, package, processors, threads, pretagged=False):
        import torch
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        if isinstance(processors, dict):  # --model: exactly StanzaSharp's models
            package = None
        self.nlp = stanza_pipeline(models, lang, package, processors, **({"pretagged": True} if pretagged else {}))
        self.pretagged = pretagged

    def __call__(self, text):
        try:
            with self.torch.no_grad():
                if self.pretagged:
                    from stanza.utils.conll import CoNLL
                    doc = self.nlp.process(CoNLL.conll2doc(input_str=text))
                else:
                    doc = self.nlp(text)
                conllu = "{:C}".format(doc)
                return {"conllu": conllu + "\n" if conllu else ""}  # as compare.py: "" without sentences
        except Exception as e:  # noqa: BLE001 - Stanza's failures are results here
            return {"error": f"{type(e).__name__}: {e}"}


class Sharp:
    def __init__(self, exe, models, lang, package, processors, backend, threads, chosen=(), pretagged=False):
        self.cmd = ([str(exe), "text-fuzz", "--models", str(Path(models) / lang), "--lang", lang, "--package", package,
                     "--processors", processors, "--backend", backend] + (["--threads", str(threads)] if threads else [])
                    + [a for m in chosen for a in ("--model", m)] + (["--pretagged"] if pretagged else []))
        self.models = None
        self.start()

    def start(self):
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     encoding="ascii", bufsize=1)
        line = self.proc.stdout.readline()
        if not line.startswith('{"ready"'):
            raise RuntimeError(f"text-fuzz did not start: {line!r}")
        self.models = json.loads(line).get("models")  # processor -> model, Stanza's order

    def __call__(self, text):
        try:
            self.proc.stdin.write(json.dumps(text) + "\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except OSError:
            line = ""
        if not line:  # the process died (stack overflow, fail-fast): a crash
            code = self.proc.wait()
            self.start()
            return {"error": f"process exited with {code}"}
        return json.loads(line)

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def code_point_offsets(conllu, text):
    """C#'s start_char/end_char are UTF-16 indices; Stanza's are code points."""
    if all(ord(c) <= 0xFFFF for c in text):
        return conllu
    to_cp, units = {}, 0
    for i, c in enumerate(text):
        to_cp[units] = i
        units += 2 if ord(c) > 0xFFFF else 1
    to_cp[units] = len(text)

    def fix(part):
        for key in ("start_char=", "end_char="):
            if part.startswith(key):
                return key + str(to_cp[int(part[len(key):])])
        return part
    lines = []
    for line in conllu.split("\n"):
        cols = line.split("\t")
        if len(cols) == 10:
            cols[9] = "|".join(fix(p) for p in cols[9].split("|"))
        lines.append("\t".join(cols))
    return "\n".join(lines)


FIELDS = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"]


def first_difference(a, b):
    """The first differing line of two CoNLL-U strings, named by its field: (kind, stanza line, C# line)."""
    al, bl = a.split("\n"), b.split("\n")
    for x, y in zip(al, bl):
        if x == y:
            continue
        if x.startswith("#") or y.startswith("#"):
            kx, ky = x.partition("=")[0], y.partition("=")[0]
            return (kx.strip("# ") if kx == ky else "comments/structure"), x, y
        cx, cy = x.split("\t"), y.split("\t")
        if len(cx) != 10 or len(cy) != 10 or cx[0] != cy[0]:
            return "tokens/sentences", x, y
        for k in range(10):
            if cx[k] != cy[k]:
                if k == 9:
                    mx, my = cx[9].split("|"), cy[9].split("|")
                    keys = [p.partition("=")[0] for p, q in zip(mx, my) if p != q] or ["MISC"]
                    return "MISC " + keys[0], x, y
                return FIELDS[k], x, y
    return "length", "\n".join(al[len(bl):])[:200], "\n".join(bl[len(al):])[:200]


SECONDS = {"stanza": 0.0, "csharp": 0.0}


def timed(side, name, text):
    clock = time.perf_counter()
    try:
        return side(text)
    finally:
        SECONDS[name] += time.perf_counter() - clock


def check(stanza, sharp, text):
    """None if both sides agree, else (kind, details)."""
    a, b = timed(stanza, "stanza", text), timed(sharp, "csharp", text)
    if "error" in b:
        return "C# error" + (" (both)" if "error" in a else ""), {"stanza": a.get("error"), "csharp": b["error"][:2000]}
    if "error" in a:
        return "Stanza error", {"stanza": a["error"], "csharp": None}
    conllu = code_point_offsets(b["conllu"], text)
    if a["conllu"] == conllu:
        return None
    kind, x, y = first_difference(a["conllu"], conllu)
    return kind, {"stanza": x, "csharp": y}


def shrink(text, still_fails, budget):
    """ddmin over code points down to single ones: the smallest text for which still_fails holds, trying at most
    budget candidates (a long text through the whole pipeline takes seconds a candidate)."""
    chars, n = list(text), 2
    while len(chars) >= 2 and budget > 0:
        size = max(1, len(chars) // n)
        for start in range(0, len(chars), size):
            candidate = chars[:start] + chars[start + size:]
            budget -= 1
            if budget < 0:
                break
            if candidate and still_fails("".join(candidate)):
                chars, n = candidate, max(n - 1, 2)
                break
        else:
            if size == 1:
                break
            n = min(n * 2, len(chars))
    return "".join(chars)


class Pretagger:
    """--pretagged: a text as the CoNLL-U document both sides annotate. Stanza's tokenize,mwt,pos,lemma (the language's
    default package) tag it; then, seeded by the text, about one word in ten loses its lemma and one in twenty its XPOS
    (never UPOS: depparse needs one of the two), as PRETAGGED_CASES strips whole fields."""

    def __init__(self, models, lang):
        import torch
        self.torch = torch
        self.nlp = stanza_pipeline(models, lang, "default", "tokenize,mwt,pos,lemma")
        self.last = (None, None)  # both sides ask for the same text in turn

    def __call__(self, text):
        if self.last[0] == text:
            return self.last[1]
        self.last = (text, self.tag(text))
        return self.last[1]

    def tag(self, text):
        with self.torch.no_grad():
            doc = self.nlp(text)
        rng = random.Random(text)
        for sentence in doc.sentences:
            for word in sentence.words:
                if rng.random() < 0.1:
                    word.lemma = None
                if rng.random() < 0.05:
                    word.xpos = None
        conllu = "{:C}".format(doc)
        return conllu + "\n" if conllu else ""


def run(args, texts):
    processors = args.processors or processors_for(args.lang, args.package)
    exe = args.benchmark or next(iter(sorted((ROOT / "samples" / "StanzaSharp.Benchmark" / "bin" / "Release").glob(
        "net*/StanzaSharp.Benchmark.exe" if sys.platform == "win32" else "net*/StanzaSharp.Benchmark"))), None)
    if exe is None:
        sys.exit("build samples/StanzaSharp.Benchmark -c Release first, or pass --benchmark")
    sharp = Sharp(exe, args.models, args.lang, args.package, processors, args.backend, args.threads, args.model,
                  args.pretagged)
    chosen = sharp.models
    if args.pretagged:  # StanzaSharp always loads tokenize and mwt; Stanza processes the document without them
        chosen = {p: m for p, m in chosen.items() if p in processors.split(",")}
    stanza = Stanza(args.models, args.lang, args.package, chosen if args.model else processors, args.threads,
                    args.pretagged)
    if args.pretagged:  # the texts become documents; the shrinking still works on the text
        tag, side_stanza, side_sharp = Pretagger(args.models, args.lang), stanza, sharp
        stanza, sharp = (lambda t: side_stanza(tag(t))), (lambda t: side_sharp(tag(t)))
    clock, found = time.perf_counter(), []
    for i, t in enumerate(texts):
        if i and i % 50 == 0:
            print(f"{i}/{len(texts)} texts, {len(found)} differences, {time.perf_counter() - clock:.0f} s", file=sys.stderr)
        result = check(stanza, sharp, t)
        if result is None:
            continue
        kind, details = result
        print(f"text {i}: {kind}: {details}", file=sys.stderr)
        small = t
        if not args.no_shrink:
            small = shrink(t, lambda c: (r := check(stanza, sharp, c)) is not None and r[0] == kind,
                           args.shrink_budget)
            print(f"  shrunk to {small!r}: {check(stanza, sharp, small)}", file=sys.stderr)
        found.append({"index": i, "kind": kind, "details": details, "text": t, "minimal": small,
                      "minimal_details": check(stanza, sharp, small)})
    (side_sharp if args.pretagged else sharp).close()
    meta = {"lang": args.lang, "package": args.package, "processors": processors, "models": args.model,
            "pretagged": args.pretagged, "seed": args.seed,
            "texts": len(texts), "characters": sum(len(t) for t in texts), "seconds": time.perf_counter() - clock,
            "differences": len(found), "backend": args.backend,
            "stanza_seconds": SECONDS["stanza"], "csharp_seconds": SECONDS["csharp"]}
    print(json.dumps(meta))
    if args.report:
        Path(args.report).write_text(json.dumps({"meta": meta, "found": found}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    return 1 if found else 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "generate", "texts"):
        s = sub.add_parser(name)
        if name == "texts":
            s.add_argument("file")
        s.add_argument("--lang", default="en")
        s.add_argument("--seed", type=int, default=1)
        s.add_argument("--count", type=int, default=200 if name == "run" else 20)
        s.add_argument("--package", default="default")
        s.add_argument("--processors")
        s.add_argument("--model", action="append", default=[], help="P=NAME: processor P's model (repeatable)")
        s.add_argument("--pretagged", action="store_true", help="annotate tagged documents (Stanza's pretagged=True)")
        s.add_argument("--words", help="extra fragments, comma-separated, spliced in about four times in ten")
        s.add_argument("--models", default=str(ROOT / "models" / "stanza"))
        s.add_argument("--benchmark", help="the built StanzaSharp.Benchmark executable")
        s.add_argument("--backend", default="managed", choices=["managed", "torch"])
        s.add_argument("--threads", type=int, default=4, help="each side's threads (0: their defaults)")
        s.add_argument("--report")
        s.add_argument("--no-shrink", action="store_true")
        s.add_argument("--shrink-budget", type=int, default=300, help="candidates tried per shrink")
    args = p.parse_args()
    WORDS.extend(w for w in (args.words or "").split(",") if w)
    if args.cmd == "generate":
        for t in generate(args.lang, args.seed, args.count):
            print(ascii(t))
        return 0
    if args.cmd == "texts":
        return run(args, json.loads(Path(args.file).read_text(encoding="utf-8")))
    return run(args, generate(args.lang, args.seed, args.count))


if __name__ == "__main__":
    sys.exit(main())
