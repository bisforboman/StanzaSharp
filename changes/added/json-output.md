- `Json.Write(doc)` writes a document as Python Stanza's `str(doc)` does (`json.dumps(doc.to_dict(), indent=2,
  ensure_ascii=False)`), byte for byte, and `Json.Read(json)` reads it back (`Json.Write(Json.Read(json))` equals
  `json`). Like Stanza's JSON, it holds the tokens and words only, not the sentence text, id, tree or sentiment.
