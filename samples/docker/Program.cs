// Reads English text on stdin and writes Stanza's CoNLL-U to stdout.
// The models directory is the first argument (the Dockerfile passes /models, where they are mounted).
using StanzaSharp;

using var nlp = Pipeline.Load(args.Length > 0 ? args[0] : "/models");
Console.Write(Conllu.Write(nlp.Process(Console.In.ReadToEnd())));
