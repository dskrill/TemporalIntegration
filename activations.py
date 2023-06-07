def preproc_stim(text, lower=False, model_name_or_path='gpt2',tokenizer_kwargs = {},tokenizer=None):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, add_special_token=False,add_prefix_space=True,**tokenizer_kwargs)

    df = pd.DataFrame({'word': text.split()})
    text = format_text(text, lower=lower)
    transcript_tokens = space_tokenizer(text)
    gentle_tokens = gentle_tokenizer(text)
    #assert len(gentle_tokens) == len(df)

    spans = match_transcript_tokens(transcript_tokens, gentle_tokens)
    assert len(spans) == len(gentle_tokens)

    tokens = [w[0] for w in spans]
    tokens = format_tokens(tokens, lower=lower)

    # word raw
    # df.set_index(keys="word", inplace=True)
    df["word_raw"] = tokens

    # is_final_word
    begin_of_sentences_marks = [".", "!", "?"]
    # df["is_eos"] = [np.any([k in i for k in begin_of_sentences_marks])
    #                 for i in tokens]

    df["is_eos"] = [np.any([k in i for k in begin_of_sentences_marks])
                    for i in df["word"].values]

    # is_bos
    df["is_bos"] = np.roll(df["is_eos"], 1)

    # seq_id
    df["sequ_index"] = df["is_bos"].cumsum() - 1

    # wordpos_in_seq
    df["wordpos_in_seq"] = df.groupby("sequ_index").cumcount()

    # wordpos_in_stim
    df["wordpos_in_stim"] = np.arange(len(tokens))

    # seq_len
    df["seq_len"] = df.groupby("sequ_index")["word_raw"].transform(len)

    # end of file
    df["is_eof"] = [False] * (len(df) - 1) + [True]
    df["is_bof"] = [True] + [False] * (len(df) - 1)

    df["word_raw"] = df["word_raw"].fillna("")
    df["word"] = df["word"].fillna("")
    df['tokens'] = [tokenizer.encode(row["word_raw"]) for i,row in df.iterrows()]

    # if 'bert' in model_name_or_path:
    #     df['tokens'] = [[t for t in row['tokens'] if t not in [101,102]] for i,row in df.iterrows()]
    
    if 'roberta' in model_name_or_path:
        df['tokens'] = [[t for t in row['tokens'] if t not in [0,2]] for i,row in df.iterrows()]
    elif 'llama' in model_name_or_path:
        df['tokens'] = [[t for t in row['tokens'] if t not in [1]] for i,row in df.iterrows()]
    
    df['n_tokens_in_word'] = [len(i) for i in df['tokens']]
    df['decoded'] = [tokenizer.convert_ids_to_tokens(row['tokens']) for i,row in df.iterrows()]
    

    return(df)