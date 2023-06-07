import pandas as pd
import torch
import numpy as np
import itertools
import itertools
import warnings
from tqdm.auto import tqdm

def text_to_df(text,tokenizer,excluded_token_ids = []):
    """ Converts text to a Pandas DataFrame and tokenizes.

    Also counts the number of tokens in each word, necessary for post processing
    words with more than one token (e.g., by taking the mean activation across tokens).

    Args:
        text (list): list of strings to be tokenized
        tokenizer: Huggingface tokenizer
        excluded_token_ids (list): list of token ids to be excluded from the analysis

    Returns:
        df (pd.DataFrame): DataFrame with the following columns:
            - word: the original word
            - tokens: the tokenized word
            - n_tokens_in_word: the number of tokens in the word
            - decoded: the decoded tokens (sanity check)
    """

    df = pd.DataFrame({'word': text})
    df['tokens'] = [tokenizer.encode(row["word"]) for i,row in df.iterrows()]
    df['tokens'] = [[t for t in row['tokens'] if t not in excluded_token_ids] for i,row in df.iterrows()]
    df['n_tokens_in_word'] = [len(i) for i in df['tokens']]
    df['decoded'] = [tokenizer.convert_ids_to_tokens(row['tokens']) for i,row in df.iterrows()]
    
    return(df)

def aggregate_tokens(df,activations):
    """Post-processes activations by calculating the mean across tokens in a word."""
    assert torch.any(torch.isnan(activations)) == False
    assert df['n_tokens_in_word'].sum() == activations.shape[1]
    out = []
    counter = 0
    for i,row in df.iterrows():
        new = activations[:,counter:counter+row['n_tokens_in_word'],:].mean(dim=1)
        out.append(new)
        counter += row['n_tokens_in_word']
    return torch.stack(out,dim=1)

def get_activations(
    dfs,
    model,
    wte_only = True,
    device='cuda'):

    """Gets activations for a list of DataFrames.

    Args:
        dfs (list): list of DataFrames
        model: Huggingface model
        wte_only (bool): whether to only use the WTE layer (only supported for GPT2 models)
        device (str): device to use for the model (e.g., 'cpu', 'cuda')
    
    Returns:
        results (torch.Tensor): tensor of activations with shape (n_layers, n_sequences, n_words, n_features)
    """

    if model.config.model_type != "gpt2" and wte_only:
        warnings.warn("wte_only is only supported for GPT2 models. Setting wte_only to False.")
        wte_only = False

    model.to(device)
    model.eval()
    results = []

    for df in dfs:
        df["word_index"] = np.arange(len(df))
        with torch.no_grad():
            inpt = torch.tensor(list(itertools.chain.from_iterable(df['tokens']))).reshape(1,-1)
            inpt = inpt.to(model.device)
            out = model(inpt, output_hidden_states=True)
            out = torch.stack(out.hidden_states)
            if wte_only:
                out[0] = model.base_model.wte.forward(inpt)[None]
            out = out.cpu()

        out = aggregate_tokens(df,out.squeeze())
        results.append(out)
    results = torch.stack(results,dim=1)
    return results

def calculate_differences(swapped_seqs,original_seqs,tokenizer,model,device='cuda'):
    """Calculates the differences for a list of swapped and original sequences.

    Args:
        swapped_seqs (list): list of lists of swapped sequences
        original_seqs (list): list of original sequences
        tokenizer: Huggingface tokenizer
        model: Huggingface model
        device (str): device to use for the model (e.g., 'cpu', 'cuda')

    Returns:
        out (torch.Tensor): tensor of differences with shape (n_layers, swap position, measured position, n_features)
    """

    out = None
    n = 0
    for zz,(swapped,original) in tqdm(enumerate(zip(swapped_seqs,original_seqs))):
        try:
            swapped_dfs = []
            for s in swapped:
                df = text_to_df(s,tokenizer=tokenizer)
                swapped_dfs.append(df)
            original_df = text_to_df(original,tokenizer=tokenizer)
            swapped_activations = get_activations(swapped_dfs,model=model,device=device)
            original_activations = get_activations([original_df],model=model,device=device)

            if (swapped_activations is not None) and (original_activations is not None):
                difference = torch.abs(swapped_activations - original_activations)
                if out is None:
                    out = difference
                else:
                    out += difference
                n += 1
        except ValueError:
            print("Error: ",zz,original)
            continue
    print("Finished calculating difference tensor for ",n," sequences")
    return out/n