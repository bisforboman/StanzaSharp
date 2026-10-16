- Tokenizer: a character outside the BMP could change tokens, since the regexes Stanza runs over code points ran over
  UTF-16 text: a letter such as `𠀁` before `C:` was no word character for the `labeled_field` feature (`𠀁C` became one
  token), and an emoji counted as two characters for a URL's tail (`example.com.😀` was kept as one token). Found by
  the new differential text fuzzer (`tools/text_fuzz.py`, docs/text-fuzzing.md).
