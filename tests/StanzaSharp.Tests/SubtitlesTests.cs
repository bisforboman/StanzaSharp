using StanzaSharp.Subtitles;

namespace StanzaSharp.Tests;

/// <summary>
/// The subtitle line-break sample (samples/StanzaSharp.Subtitles, #267): SRT reading and the reassembled transcript,
/// the rule table and break scoring on hand-made parses, and the English example end to end.
/// </summary>
public class SubtitlesTests(SharedPipelines shared) : IClassFixture<SharedPipelines>
{
    private const string Srt2 = "﻿1\r\n00:00:01,000 --> 00:00:02,000\r\n<i>We drove to the</i>\r\nold house.\r\n\r\n"
        + "2\r\n00:00:03,000 --> 00:00:04,000\r\n- Who?\r\n- {\\an8}Her.\r\n";

    [Fact]
    public void Parse_ReadsCuesWithTheirLines()
    {
        var cues = Srt.Parse(Srt2);
        Assert.Equal(2, cues.Count);
        Assert.Equal(1, cues[0].Number);
        Assert.Equal("00:00:01,000 --> 00:00:02,000", cues[0].Timing);
        Assert.Equal(["<i>We drove to the</i>", "old house."], cues[0].Lines);
        Assert.Equal(["- Who?", "- {\\an8}Her."], cues[1].Lines);
    }

    [Fact]
    public void Transcript_JoinsLinesWithoutTagsOrDashes()
    {
        var transcript = Transcript.Build(Srt.Parse(Srt2));
        Assert.Equal("We drove to the old house. Who? Her.", transcript.Text);
        Assert.Equal(16, transcript.Lines[0][1].Start);
        Assert.Equal([false, false], transcript.Lines[0].Select(l => l.Dialogue));
        Assert.Equal([true, true], transcript.Lines[1].Select(l => l.Dialogue));
        Assert.Equal("Her.", transcript.Lines[1][1].Text);
    }

    // "We drove to the | old house.": the break separates "the" (det) and "to" (case) from "house".
    private const string Parse = """
        1	We	we	PRON	_	_	2	nsubj	_	start_char=0|end_char=2
        2	drove	drive	VERB	_	_	0	root	_	start_char=3|end_char=8
        3	to	to	ADP	_	_	6	case	_	start_char=9|end_char=11
        4	the	the	DET	_	_	6	det	_	start_char=12|end_char=15
        5	old	old	ADJ	_	_	6	amod	_	start_char=16|end_char=19
        6	house	house	NOUN	_	_	2	obl	_	start_char=20|end_char=25|SpaceAfter=No
        7	.	.	PUNCT	_	_	2	punct	_	start_char=25|end_char=26

        """;

    [Fact]
    public void Check_FlagsTheRelationsTheBreakSplits_AndSuggestsABreakBeforeThePreposition()
    {
        var cues = Srt.Parse("1\n00:00:01,000 --> 00:00:02,000\nWe drove to the\nold house.\n");
        var finding = Assert.Single(BreakChecker.Check(Transcript.Build(cues), Conllu.Read(Parse)));
        Assert.Equal(1, finding.Line);
        Assert.Equal(["case to house", "det the house"], finding.Splits.Select(s => $"{s.Relation} {s.Dependent} {s.Head}"));
        Assert.True(finding.FitsOneLine);
        Assert.NotNull(finding.Suggestion);
        Assert.Equal(("We drove", "to the old house."), (finding.Suggestion.Top, finding.Suggestion.Bottom));
        Assert.Empty(finding.Suggestion.Splits);
    }

    [Fact]
    public void Check_PassesABreakThatSplitsNothing()
    {
        var cues = Srt.Parse("1\n00:00:01,000 --> 00:00:02,000\nWe drove\nto the old house.\n");
        Assert.Empty(BreakChecker.Check(Transcript.Build(cues), Conllu.Read(Parse)));
    }

    [Fact]
    public void Check_SkipsTheBreakBeforeADialogueLine()
    {
        var cues = Srt.Parse("1\n00:00:01,000 --> 00:00:02,000\n-We drove to the\n-old house.\n");
        Assert.Empty(BreakChecker.Check(Transcript.Build(cues), Conllu.Read(Parse)));
    }

    [ModelFact]
    public void Example_English_FlagsTheBadBreaks()
    {
        var nlp = shared.Get(processors: "tokenize,pos,lemma,depparse");
        var transcript = Transcript.Build(Srt.Parse(File.ReadAllText(Path.Combine(Repo.Root, "samples", "StanzaSharp.Subtitles", "example.en.srt"))));
        var findings = BreakChecker.Check(transcript, nlp.Process(transcript.Text));
        Assert.Equal(
            ["1 case det", "2 mark", "3 flat", "4 compound:prt", "6 det", "9 nsubj"],
            findings.Select(f => $"{f.Cue.Number} {string.Join(" ", f.Splits.Select(s => s.Relation))}"));
        Assert.Equal(("Nobody had lived there", "since Margaret left in the spring."), (findings[1].Suggestion!.Top, findings[1].Suggestion!.Bottom));
    }
}
