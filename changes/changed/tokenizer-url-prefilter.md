- The tokenizer skips Stanza's e-mail/URL pattern on paragraphs that contain none of the literals every match needs
  (`@`, `://`, `www.`, `.gov`, `.org`, `.edu`, `.net`, `.co`). The output is unchanged.
