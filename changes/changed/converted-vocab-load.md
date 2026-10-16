- Converted models (`tools/stanza_convert.py`'s `.json` + `.safetensors`) load 15-25% faster: their large
  vocabularies are read straight into lookup tables, as the `.pt` files' have been since 1.5.0, instead of through JSON
  nodes.
