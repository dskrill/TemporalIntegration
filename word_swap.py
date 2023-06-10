import torch
import pandas as pd
import numpy as np
import os 
import regex
import pickle
import spacy
import random
import itertools
nlp = spacy.load("en_core_web_sm")
import nltk
from tqdm.autonotebook import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

def create_distribution(word,tokenizer,candidate_embeddings,mean,half_width,embedding_function, excluded_token_ids = [], size=10,spacing=.001):
    out = None
    with torch.no_grad():
        token = tokenizer.encode(word)
        token = torch.tensor([t for t in token if t not in excluded_token_ids])

        target = embedding_function(token).mean(0,keepdim=True)

        assert target.ndim == 2
        diff = torch.abs(candidate_embeddings - target).mean(-1)
        for c in torch.arange(mean-half_width,mean+half_width,spacing):
            indices = torch.nonzero((diff > c) & (diff < c+spacing),as_tuple=True)[0]
            ridx = np.random.choice(len(indices),size=size,replace=True)
            if out is None:
                out = indices[ridx]
            else:
                out = torch.cat([out,indices[ridx]])
        return out

def create_sequences(segments,n_context = 2):
    temp = [" ".join(s) for s in segments]
    sequences = []
    for s in segments:
        before = None
        after = None
        for i in range(n_context):
            if before is None:
                before = np.random.choice(temp,1)[0].split()
                after = np.random.choice(temp,1)[0].split()
            else:
                before += np.random.choice(temp,1)[0].split()
                after += np.random.choice(temp,1)[0].split()

        sequences.append(before + s + after)
    assert len(np.unique([len(s) for s in sequences])) == 1
    return sequences


def filter_word_list(word_list,tokenizer,excluded_token_ids = []):
    wlist = [w for w in word_list if (w.isalpha())]
    out = []
    tokens = []
    for word in tqdm(wlist):
        current = tokenizer.encode(word)
        current = torch.tensor([q for q in current if q not in excluded_token_ids])
        if len(current) == 1:
            out.append(word)
            tokens.append(current)
    return out,tokens

def create_yoking_sequences(sentences, tokenizer,excluded_token_ids,n_context = 2,n_sentences=250):
        sentences_ = []
        sentence_len = len(sentences[0])
        for s in sentences:
            if ("".join(s).isalpha()):
                tokenized = [x for x in tokenizer.encode(" ".join(s)) if x not in excluded_token_ids]

                if len(tokenized) == sentence_len:
                    sentences_.append(s)


        sentences_ = [s for s in sentences if ("".join(s).isalpha()) and (len([x for x in tokenizer.encode(" ".join(s)) if x not in excluded_token_ids]) == sentence_len)]
        sentences_ = random.choices(sentences_,k=n_sentences)
        sequences = create_sequences(sentences_,n_context = n_context)
        return sequences

# word_list,tokens = filter_word_list(word_list)

def create_candidate_embeddings(tokens,embedding_function):
    with torch.no_grad():
        candidate_embeddings = embedding_function(torch.stack(tokens)).squeeze()
    print("Candidate embeddings created of shape: ",candidate_embeddings.shape)
    return candidate_embeddings

class WordSwap:
    def __init__(self,excluded_token_ids=[]):
        self.original_sequences = []
        self.swapped = []
        self.excluded_token_ids = excluded_token_ids 

        self.ok = True

    def __call__(self,sequences):
        pass

class RandomWordSwap(WordSwap):
    def __init__(self,excluded_token_ids=[]):
        super().__init__(excluded_token_ids=excluded_token_ids)

    def __repr__(self):
        return "Word swaps using random choice from word list"

    def __call__(self,sequences):
        n_sequences = len(sequences)

        for sequence_num,s in tqdm(enumerate(sequences)):
            other = sequences[np.random.choice(range(n_sequences),1)[0]]
            swapped_sequences = []
            for i,word in enumerate(s):
                temp = s.copy()
                to_swap = random.choice(word_list)
                while to_swap == word:
                    to_swap = random.choice(word_list)
                temp[i] = to_swap
                swapped_sequences.append(temp)

            self.swapped.append(swapped_sequences)
            self.original_sequences.append(s)

class RandomPosWordSwap(WordSwap):
    def __init__(self,word_lookup,pos_dict,tokenizer,excluded_token_ids = []):
        super().__init__(excluded_token_ids=excluded_token_ids)
        self.word_lookup = word_lookup
        self.pos_dict = pos_dict
        self.tokenizer = tokenizer
        self.word_list = []
        for w in list(word_lookup.keys()):
            tokens = [t for t in tokenizer.encode(w) if t not in excluded_token_ids]
            if len(tokens) == 1:
                self.word_list.append(w)
    
    def __repr__(self):
        return "Word swaps using random choice from same POS"

    def __call__(self,sequences):
        for sequence_num,s in tqdm(enumerate(sequences)):
            swapped_sequences = []
            for i,word in enumerate(s):
                temp = s.copy()
                current_pos = self.word_lookup[word]
                if current_pos not in self.pos_dict:
                    print("Warning: ",current_pos," not in pos_dict")
                    print(word)
                    to_swap = random.choice(self.word_list)
                else:
                    to_swap = random.choice(self.pos_dict[current_pos])
                try_count = 0
                token = self.tokenizer.encode(to_swap)
                token = [t for t in token if t not in self.excluded_token_ids]
                while (to_swap == word) or len(token)>1:
                    if (try_count < 20) and (current_pos in self.pos_dict):
                        to_swap = random.choice(self.pos_dict[current_pos]) # some rare pos tags may not have many words
                        token = self.tokenizer.encode(to_swap)
                        token = [t for t in token if t not in self.excluded_token_ids]
                        try_count += 1
                        # print(try_count,current_pos)
                        if try_count == 20:
                            print(f"Warning: {current_pos} has few words (current word: {word})")
                            break
                    else:
                        to_swap = random.choice(self.word_list) # just pick randomly in that case
                        break
                temp[i] = to_swap
                swapped_sequences.append(temp)
            self.swapped.append(swapped_sequences)
            self.original_sequences.append(s)



class DistributionWordSwap(WordSwap):
    def __init__(self,word_list,candidate_embeddings,embedding_function,sampling_params,tokenizer,excluded_token_ids = []):
        super().__init__(excluded_token_ids=excluded_token_ids)
        self.word_list = word_list
        self.candidate_embeddings = candidate_embeddings
        self.embedding_function = embedding_function
        self.sampling_params = sampling_params
        self.tokenizer = tokenizer
        assert candidate_embeddings.shape[0] == len(word_list), "Word list and candidate embeddings don't match"


    def __repr__(self):
        return "Word swaps with embedding differences sampled from a uniform distribution"

    def __call__(self,sequences):
        potential_swaps = {}
        all_diffs = {}
        for sequence_num,s in tqdm(enumerate(sequences)):
            swapped_sequences = []
            for i,word in enumerate(s):
                if self.ok:
                    temp = s.copy()
                    try:
                        distribution = create_distribution(word,
                                                            self.tokenizer,
                                                            self.candidate_embeddings,
                                                            **self.sampling_params,
                                                            embedding_function = self.embedding_function,
                                                            excluded_token_ids = self.excluded_token_ids)

                        selected_idx = np.random.choice(distribution)
                        temp[i] = self.word_list[selected_idx]
                    except:
                        print("Using random word sub for: ",word)
                        temp[i] = random.choice(self.word_list)
                    swapped_sequences.append(temp)
            self.swapped.append(swapped_sequences)
            self.original_sequences.append(s)



class ProbableWordSwap(WordSwap):
    def __init__(self,unmasker,idx_low = 0,idx_high = 100,sentence_len = None,excluded_token_ids = []):
        super().__init__(excluded_token_ids)
        self.unmasker = unmasker
        self.idx_low = idx_low
        self.idx_high = idx_high
        self.sentence_len = sentence_len


    def __repr__(self):
        return "Word swaps sampled from a probable pool of words (determined using a user-specified unmasker)"

    def __call__(self,sequences):
        for sequence_num,s in tqdm(enumerate(sequences)):
            try:
                swapped_sequences = []
                for i,word in enumerate(s):
                    temp = s.copy()
                    temp[i] = '[MASK]'
                    if self.sentence_len:
                        current_sentence_start = (i//self.sentence_len)*self.sentence_len
                        current_sentence_end = current_sentence_start + self.sentence_len
                        current_sentence = " ".join(temp[current_sentence_start:current_sentence_end])
                    else:
                        current_sentence = " ".join(temp)
                    
                    potential_swaps = np.array([x["token_str"] for x in self.unmasker(current_sentence) if ((x["token_str"] != word) and x["token_str"].isalpha())])
                    n_potential_swaps = len(potential_swaps)
                    temp[i] = potential_swaps[np.random.randint(low=self.idx_low,high=min(self.idx_high,n_potential_swaps),size=1)[0]] 
                    swapped_sequences.append(temp)
            except Exception as e:
                self.ok = False
                print(e)
                continue

            if self.ok:
                self.swapped.append(swapped_sequences)
                self.original_sequences.append(s)



# def word_swap(sequences,control = "distribution"):
#     n_sequences = len(sequences)
#     swapped = []
#     original_sequences = []

#     for sequence_num,s in tqdm(enumerate(sequences)):
#         other = sequences[np.random.choice(range(n_sequences),1)[0]]
#         swapped_sequences = []
        
#         if control == "random":
#             ok = True
#             word_pool = list(itertools.chain.from_iterable(sequences))
#             try:
#                 for i,word in enumerate(s):
#                     temp = s.copy()
#                     to_swap = random.choice(word_pool)
#                     while to_swap == word:
#                         to_swap = random.choice(word_pool)
#                     temp[i] = to_swap
#                     swapped_sequences.append(temp)
                
#             except:
#                 ok = False
#                 continue

#         if control == "random_pos":
#             ok = True

#             for i,word in enumerate(s):
#                 temp = s.copy()
#                 current_pos = word_lookup[word]
#                 if current_pos not in pos_dict:
#                     print("Warning: ",current_pos," not in pos_dict")
#                     to_swap = random.choice(word_list)
#                 else:
#                     to_swap = random.choice(pos_dict[current_pos])
#                 try_count = 0
#                 token = tokenizer.encode(to_swap)
#                 token = [t for t in token if t not in excluded_token_ids]
#                 while (to_swap == word) or len(token)>1:
#                     if (try_count < 10) and (current_pos in pos_dict):
#                         to_swap = random.choice(pos_dict[current_pos]) # some rare pos tags may not have many words
#                         token = tokenizer.encode(to_swap)
#                         token = [t for t in token if t not in excluded_token_ids]
#                         try_count += 1
#                         if try_count == 10:
#                             print(f"Warning: {current_pos} has few words (current word: {word})")
#                             break
#                     else:
#                         to_swap = random.choice(word_list) # just pick randomly in that case
#                 temp[i] = to_swap
#                 swapped_sequences.append(temp)

#         elif control == "distribution":
#             potential_swaps = {}
#             all_diffs = {}
#             assert candidate_embeddings.shape[0] == len(word_list), "Word list and candidate embeddings don't match"

#             ok = True
#             for i,word in enumerate(s):
#                 if ok:
#                     # try:
#                     temp = s.copy()
#                     try:
#                         distribution = create_distribution(word, candidate_embeddings,size=100,**sampling_params[model.config.model_type])

#                         selected_idx = np.random.choice(distribution)
#                         temp[i] = word_list[selected_idx]
#                     except:
#                         print("Using random word sub for: ",word)
#                         temp[i] = random.choice(word_list)
#                     swapped_sequences.append(temp)

#         elif (control == "probable") or (control == "improbable"):
#             ok = True
#             try:
#                 for i,word in enumerate(s):
#                     temp = s.copy()
#                     temp[i] = '[MASK]'
#                     if not analysis == "overall_integration":
#                         current_sentence_start = (i//sentence_len)*sentence_len
#                         current_sentence_end = current_sentence_start + sentence_len
#                         current_sentence = " ".join(temp[current_sentence_start:current_sentence_end])
#                     else:
#                         current_sentence = " ".join(temp)
                    
#                     potential_swaps = np.array([x["token_str"] for x in unmasker(current_sentence) if ((x["token_str"] != word) and x["token_str"].isalpha())])
#                     # potential_swaps[i] = [o for o in out if nlp(o.item())[0].pos_ == nlp(word)[0].pos_]
#                     n_potential_swaps = len(potential_swaps)
#                     if control == "probable":
#                         temp[i] = potential_swaps[np.random.randint(low=0,high=min(100,n_potential_swaps),size=1)[0]] 
#                     elif control == "improbable":
#                         temp[i] = potential_swaps[np.random.randint(low=max(0,n_potential_swaps-100),high=n_potential_swaps,size=1)[0]]
#                     else:
#                         raise ValueError("Invalid control")
#                     swapped_sequences.append(temp)
#             except Exception as e:
#                 ok = False
#                 print(e)
#                 continue

#             # with torch.no_grad():
#             #     target = model.base_model.wte.forward(tokenizer.encode(word,return_tensors="pt")).mean(1,keepdim=True)
#             #     tokens = tokenizer.encode(" ".join(potential_swaps[i]),return_tensors="pt")
#             #     candidate_embeddings  = model.base_model.wte.forward(tokens)
#             #     # need embeddings to be at word level, not token?
#             # differences = torch.abs(target - candidate_embeddings)
#             # all_diffs[i] = differences.mean(-1).squeeze().ravel()
#             # resampled_indices = histogram_equalization(all_diffs[i], marginal.ravel())
#             # resampled_candidates = tokens[...,resampled_indices].squeeze()
#             # selected_swap = resampled_candidates[np.random.choice(len(resampled_candidates),1)[0]]
#             # temp[i] = tokenizer.decode(selected_swap).strip()
#             # swapped_sequences.append(temp)
#             # ok = True

#         if ok:
#             swapped.append(swapped_sequences)
#             original_sequences.append(s)

#     return original_sequences,swapped

# originals,swapped = word_swap(sequences,control=control)
