using System.Text.Json.Nodes;

namespace StanzaSharp.Nn;

/// <summary>
/// A tagger's or parser's XPOS vocabulary: Stanza's <c>WordVocab</c> (one tag), or an <c>XPOSVocab</c> (a CompositeVocab, not
/// keyed: each tag is parts joined by <c>sep</c>, English lines "-", partut "" for one part per character), whose parts the
/// tagger predicts and the parser embeds one by one.
/// </summary>
internal sealed class XposVocab
{
    private const int UnkId = 1, EmptyId = 2; // vocab.UNK_ID, EMPTY_ID

    private readonly string[][] _idToUnit;
    private readonly Dictionary<string, int>[] _unitToId;
    private readonly string? _sep;
    private readonly bool _ignoreBlank;

    private XposVocab(string[][] idToUnit, string? sep, bool ignoreBlank)
    {
        _idToUnit = idToUnit;
        _unitToId = [.. idToUnit.Select(units => units.Select((u, i) => (u, i)).DistinctBy(x => x.u).ToDictionary(x => x.u, x => x.i))];
        (_sep, _ignoreBlank) = (sep, ignoreBlank);
    }

    /// <summary>Whether this is a composite vocabulary (the model has a head or an embedding per part).</summary>
    public bool IsComposite => _sep != null;

    /// <summary>Each part's size: one for a <c>WordVocab</c>.</summary>
    public int[] Sizes => [.. _idToUnit.Select(u => u.Length)];

    /// <summary>Reads <c>vocab["xpos"]</c> of a checkpoint's vocabulary (MultiVocab state with <c>_key2class</c>).</summary>
    public static XposVocab Read(JsonNode vocab)
    {
        var node = vocab["xpos"]!;
        string[] Strings(JsonNode? array) => [.. array!.AsArray().Select(x => x!.GetValue<string>())];
        switch (vocab["_key2class"]!["xpos"]!.GetValue<string>())
        {
            case "WordVocab":
                return new([Strings(node["_id2unit"])], null, node["ignore"]?.AsArray().Any(x => x!.GetValue<string>() == "_") == true);
            case "XPOSVocab" when node["keyed"]?.GetValue<bool>() != true:
                // _id2unit is {part index: units}, a dict with int keys: {"$dict": [[0, [...]], [1, [...]], ...]}.
                var parts = node["_id2unit"]!["$dict"]!.AsArray().Select(pair => Strings(pair![1])).ToArray();
                return new(parts, node["sep"]!.GetValue<string>(), false);
            default:
                throw new NotSupportedException($"XPOS vocabulary {vocab["_key2class"]!["xpos"]} (keyed) is not ported");
        }
    }

    /// <summary>unit2id: one id per part (UNK for an unknown part, EMPTY past the tag's parts; a WordVocab maps an ignored "_" to EMPTY).</summary>
    public void Ids(string unit, Span<long> ids)
    {
        if (!IsComposite)
        {
            ids[0] = _ignoreBlank && unit == "_" ? EmptyId : _unitToId[0].GetValueOrDefault(unit, UnkId);
            return;
        }

        // CompositeVocab.unit2parts: the characters (code points) without sep, else split on it; "_" has no parts.
        string[] parts = unit == "_" ? [] : _sep!.Length == 0 ? [.. unit.EnumerateRunes().Select(r => r.ToString())] : unit.Split(_sep);
        for (int i = 0; i < _idToUnit.Length; i++)
            ids[i] = i < parts.Length ? _unitToId[i].GetValueOrDefault(parts[i], UnkId) : EmptyId;
    }

    /// <summary>id2unit: the tag of one id per part, as Word.xpos stores it: null for "_" (a WordVocab's EMPTY, or no part).</summary>
    public string? Unit(ReadOnlySpan<long> ids)
    {
        if (!IsComposite)
        {
            var unit = _ignoreBlank && ids[0] == EmptyId ? "_" : _idToUnit[0][ids[0]];
            return unit == "_" ? null : unit;
        }

        var items = new List<string>(_idToUnit.Length);
        for (int i = 0; i < _idToUnit.Length; i++)
        {
            if (ids[i] != EmptyId)
                items.Add(_idToUnit[i][ids[i]]);
        }

        var joined = string.Join(_sep, items);
        return joined is "" or "_" ? null : joined;
    }
}
