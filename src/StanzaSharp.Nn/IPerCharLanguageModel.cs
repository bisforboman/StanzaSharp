namespace StanzaSharp.Nn;

/// <summary>
/// A character language model's state at every character (char_model.py <c>per_char_representation</c>, through
/// <c>CharacterLanguageModelWordAdapter</c>): the input the <c>_charlm</c> tokenizers (forward charlm) and lemmatizers (both)
/// add to their own. Per backend: <c>CharLanguageModel</c> or <see cref="Managed.ManagedCharLanguageModel"/>.
/// </summary>
internal interface IPerCharLanguageModel
{
    int HiddenDim { get; }

    /// <summary>The vocabulary id of one unit (a code point, or a special unit such as <c>&lt;PAD&gt;</c>); unknown → <c>&lt;UNK&gt;</c>.</summary>
    int Id(ReadOnlySpan<char> unit);

    /// <summary>
    /// Runs each sequence of ids from (h_init, c_init), left to right whatever the model's direction (per_char_representation
    /// has no backward branch) and with no start character of its own, and writes the state after position t of sequence i
    /// to row <paramref name="firstRow"/>[i] + t of <paramref name="output"/> (stride <paramref name="ldo"/> ≥ HiddenDim).
    /// Sequences must not be empty. <paramref name="ct"/> is checked before each step.
    /// </summary>
    void PerChar(IReadOnlyList<int[]> sequences, int[] firstRow, Span<float> output, int ldo, CancellationToken ct);
}
