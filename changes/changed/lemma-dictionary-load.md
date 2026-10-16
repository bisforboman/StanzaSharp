- German `Pipeline.Load` is about 15% faster: the lemmatizer keeps its 524,460-entry dictionary as the JSON's bytes
  with a hash table over them, instead of half a million strings (English unchanged).
