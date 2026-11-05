- English coreference: Stanza's three other English coref models on roberta-large, chosen with `PipelineOptions.Models`
  (`[Processor.Coref] = "gum-nospeakers_roberta-large-lora"`, `"gum-speakers_roberta-large-lora"` or
  `"ontonotes-singletons_roberta-large-lora"`; the tool's `--model coref=...`). Downloads fetch roberta-large (1.4 GB) with
  them. StanzaSharp has no speakers, so gum-speakers sees one speaker throughout, as Stanza does without its tokenizer's
  `speaker_delim`. On the TorchSharp backend two decisions in the golden texts differ from Stanza's, both near-ties of the
  model's own (roberta-large's near-identical vectors for "."; docs/verification.md).
