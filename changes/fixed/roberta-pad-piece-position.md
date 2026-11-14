- RoBERTa and XLM-RoBERTa transformers (both backends): a sentence holding the text `<pad>` gave every later piece the
  wrong position row, so its hidden states differed from transformers'. The pad piece now has row `pad_token_id` and is
  not counted, as in transformers' `create_position_ids_from_input_ids`.
