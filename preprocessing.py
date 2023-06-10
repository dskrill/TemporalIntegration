import nltk
from nltk.corpus import brown
import string
import pickle
import spacy
from tqdm.autonotebook import tqdm
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

nlp = spacy.load("en_core_web_sm")



# For conveniently removing punctuation
translator = str.maketrans('', '', string.punctuation)
excluded_pos = ["NP","NNP","NP-TL","NP$"]

# def process_sentence(s,single_token_words=False):
#     """Preprocess a sentence by removing punctuation and lowercasing.
    
#     Additionally filters sentences that contain proper nouns or 
#     non-alphabetic characters.
    
#     Args:
#         s: a list of strings (e.g., ["I", "like", "apples.")
        
#     Returns:
#         A list of words (e.g., ["i", "like", "apples"]) or None if
#         the sentence should be filtered out.
#     """

#     tags = [tag for (word,tag) in nltk.pos_tag(s)]
#     # check for proper nouns
    
#     if any(tag in excluded_pos for tag in tags):
#         return None
#     else:
#         # separate hypenated words and remove quotes
#         new = " ".join(s).replace("-"," ").replace("''","").lower()
#         # remove punctuation
#         split = new.translate(translator).split()
#         split = [s for s in split if s != "''"]
#         # check if all words are alphabetic
#         if "".join(split).isalpha():
#             return split
#         else:
#             return None

def process_sentence(s,single_token_words=False,tokenizer=None, excluded_token_ids=[]):
    """Preprocess a sentence by removing punctuation and lowercasing.
    
    Additionally filters sentences that contain proper nouns or 
    non-alphabetic characters.
    
    Args:
        s: a list of strings (e.g., ["I", "like", "apples."])
        single_token_words: if True, only add sentences that are exactly sentence_len tokens long
        tokenizer: a tokenizer object that has an encode method. If single_token_words is True, this must be provided.
        excluded_token_ids: a list of token ids to exclude from the sentence. 
        
    Returns:
        A list of words (e.g., ["i", "like", "apples"]) or None if
        the sentence should be filtered out.
    """

    assert not (single_token_words and tokenizer is None), "Must provide a tokenizer if single_token_words is True"

    tags = [tag for (word,tag) in nltk.pos_tag(s)]
    # check for proper nouns
    
    if any(tag in excluded_pos for tag in tags):
        return None
    else:
        # separate hypenated words and remove quotes
        new = " ".join(s).replace("-"," ").replace("''","").lower()
        # remove punctuation
        split = new.translate(translator).split()
        split = [s for s in split if s != "''"]
        # check if all words are alphabetic
        if "".join(split).isalpha():
            if single_token_words:
                # only add sentences that are exactly sentence_len tokens long
                tokenized = [x for x in tokenizer.encode(" ".join(split)) if x not in excluded_token_ids]
                if len(tokenized) == len(split):
                    return split
                else:
                    return None
            return split
        else:
            return None
            
# temp = []
# for s in tqdm(nltk.corpus.brown.sents()):
#     new = process_sentence(s)
#     if new is not None:
#         temp.append(new)
        
# selected_sentences = {i:[s for s in temp if len(s) == i] for i in [4,8,12,16,24,32,36]}
# counter = 0
# natural_sequence_lens = [30,35,40,45,50,55,60]
# natural_sequences = {k: [] for k in natural_sequence_lens}
# word_list = []
# pos_dict = {}
# word_lookup = {}
# for word,tag in nltk.corpus.brown.tagged_words():
#     w = word.replace("-","").replace("''","").lower().translate(translator)
#     if w.isalpha() and w != '' and not (tag in excluded_pos):
#         word_list.append(w)
#         if tag in pos_dict:
#             pos_dict[tag].append(w)
#         else:
#             pos_dict[tag] = [w]
#         word_lookup[w] = tag
#         counter += 1
#         if counter == natural_sequence_len:
#             natural_sequences[counter].append(word_list[-natural_sequence_len:])
#             # counter = 0
#     else:
#         counter = 0


# def extract_natural_sequences(tagged_words,sequence_len):
#     """Extracts natural sequences of specified length from a list of pos-tagged words. 

#     Additionally preprocesses the words by removing punctuation and lowercasing, and builds
#     up a dictionary of words to their POS tags.

#     Args:
#         tagged_words: a list of tuples (word,tag)
#         sequence_len: the length of the sequences to extract

#     Returns:    
#         a list of natural sequences of the specified length
#         a dictionary of words to their POS tags
#         a dictionary of POS tags to the words that have that tag

#     """

#     natural_sequences = []
#     word_list = []
#     pos_dict = {}
#     word_lookup = {}
#     counter = 0

#     tagged_words_no_punct = []
#     for w,t in tagged_words:
#         wp = w.translate(translator)
#         if wp.isalpha():
#             tagged_words_no_punct.append((wp,t))
#     for word,tag in tagged_words_no_punct:
#         w = word.replace("-","").replace("''","").lower().translate(translator)
#         if w.isalpha() and w != '' and not (tag in excluded_pos):
#             word_list.append(w)
#             if tag in pos_dict:
#                 pos_dict[tag].append(w)
#             else:
#                 pos_dict[tag] = [w]
#             word_lookup[w] = tag
#             counter += 1
#             if counter == sequence_len:
#                 natural_sequences.append(word_list[-sequence_len:])
#                 counter = 0
#         else:
#             counter = 0
    
#     return natural_sequences,word_lookup,pos_dict

def extract_natural_sequences(tagged_words,sequence_len):
    """Extracts natural sequences of specified length from a list of pos-tagged words. 

    Additionally preprocesses the words by removing punctuation and lowercasing, and builds
    up a dictionary of words to their POS tags.

    Args:
        tagged_words: a list of tuples (word,tag)
        sequence_len: the length of the sequences to extract

    Returns:    
        a list of natural sequences of the specified length

    """

    natural_sequences = []
    word_list = []
    counter = 0

    for word,tag in tagged_words:
        w = word.replace("-","").replace("''","").lower().translate(translator)
        if w.isalpha() and w != '' and not (tag in excluded_pos):
            word_list.append(w)
            counter += 1
            if counter == sequence_len:
                natural_sequences.append(word_list[-sequence_len:])
                counter = 0
        else:
            counter = 0
    
    return natural_sequences

def extract_noun_phrases(sentences,phrase_length):
    """Extracts noun phrases of specified length from a list of sentences.

    Args:
        sentences: a list of sentences, where each sentence is a list of words
        phrase_length: the length of the noun phrases to extract

    Returns:
        a list of noun phrases of the specified length
    """

    all_noun_phrases = []

    for sentence in sentences:
        doc = nlp(" ".join(sentence))
        all_noun_phrases.extend([chunk.text for chunk in doc.noun_chunks])
    noun_phrases = [x.split() for x in all_noun_phrases if len(x.split()) == phrase_length]
    return noun_phrases


class Corpus:
    """A class for conveniently working with corpora.

    Attributes:
        corpus: the corpus to work with, as a nltk corpus object.
        single_token_words: whether filter out words/sentences comprised of more than one token.
        tokenizer: a tokenizer to use to tokenize sentences.
        excluded_token_ids: a list of token ids to exclude from the corpus.
        all_preprocessed_sentences: a list of all preprocessed sentences in the corpus.
        pos_dict: a dictionary of POS tags to the words that have that tag.
        word_lookup: a dictionary of words to their POS tags.
    """

    def __init__(self,nltk_corpus,single_token_words=False,tokenizer=None,excluded_token_ids=[]):
        """Initialize, preprocessing sentences in the corpus and building POS and word-lookup dictionaries.

        Args:
            nltk_corpus: the corpus to work with, as a nltk corpus object. Must have a sents() method and a tagged_words() method.
            single_token_words: whether filter out words/sentences comprised of more than one token. If True, a tokenizer must be provided.
            tokenizer: a tokenizer to use to tokenize sentences. If single_token_words is True, this must be provided.
            excluded_token_ids: a list of token ids to exclude from the corpus.
        """
        self.corpus = nltk_corpus
        self.single_token_words = single_token_words
        self.tokenizer = tokenizer
        self.excluded_token_ids = excluded_token_ids
        self.all_preprocessed_sentences = []

        assert not (single_token_words and tokenizer is None), "If single_token_words is True, a tokenizer must be provided."
        if tokenizer:
            assert tokenizer.add_prefix_space, "The tokenizer must have add_prefix_space set to True."

        for s in tqdm(nltk_corpus.sents()):
            new = process_sentence(s,self.single_token_words,self.tokenizer,self.excluded_token_ids)
            if new is not None:
                self.all_preprocessed_sentences.append(new)
        self.word_lookup = {}
        self.pos_dict = {}
        self.tagged_words_no_punct = []
        for w,t in self.corpus.tagged_words():
            wp = w.translate(translator)
            if wp.isalpha():
                self.tagged_words_no_punct.append((wp,t))
        for word,tag in tqdm(self.tagged_words_no_punct):
            w = word.replace("-","").replace("''","").lower().translate(translator)
            if w.isalpha() and w != '' and not (tag in excluded_pos):
                if tag in self.pos_dict:
                    self.pos_dict[tag].append(w)
                else:
                    self.pos_dict[tag] = [w]
                self.word_lookup[w] = tag

        temp_pos_dict = {'OTHER':[]}
        for key,value in self.pos_dict.items():
            new = []
            for v in set(value):
                if self.single_token_words:
                    token = [t for t in self.tokenizer.encode(v) if t not in self.excluded_token_ids]
                    if len(token) == 1:
                        new.append(v)
                else:
                    new.append(v)

            if len(set(new)) < 5:
                temp_pos_dict['OTHER'].extend(set(new))
            else:
                temp_pos_dict[key] = new

        temp_word_lookup = {}
        for pos in temp_pos_dict:
            for word in temp_pos_dict[pos]:
                temp_word_lookup[word] = pos
        self.pos_dict = temp_pos_dict
        self.word_lookup = temp_word_lookup

        # if self.single_token_words:
        #     temp_pos_dict = {'OTHER':[]}
        #     for key,value in self.pos_dict.items():
        #         new = []
        #         for v in set(value):
        #             token = [t for t in self.tokenizer.encode(v) if t not in self.excluded_token_ids]
        #             if len(token) == 1:
        #                 new.append(v)

        #         if len(set(new)) < 5:
        #             temp_pos_dict['OTHER'].extend(set(new))
        #         else:
        #             temp_pos_dict[key] = new

        #     temp_word_lookup = {}
        #     for pos in temp_pos_dict:
        #         for word in temp_pos_dict[pos]:
        #             temp_word_lookup[word] = pos
        #     self.pos_dict = temp_pos_dict
        #     self.word_lookup = temp_word_lookup

    def get_sentences_of_length(self, length):
        """Gets all (preprocessed) sentences of a specified length."""
        return [s for s in self.all_preprocessed_sentences if len(s) == length]

    def get_natural_sequences_of_length(self, length):
        """Gets all natural sequences of a specified length."""
        natural_sequences = extract_natural_sequences(
            tagged_words = self.corpus.tagged_words(),
            sequence_len = length
        )
        return natural_sequences

    def get_noun_phrases_of_length(self, length):
        """Gets all noun phrases of a specified length."""
        noun_phrases = extract_noun_phrases(
            self.all_preprocessed_sentences,
            phrase_length = length)
        return noun_phrases


# for sentence_len in selected_sentences:
#     selected_sentences[sentence_len] = [s for s in selected_sentences[sentence_len] if all([si in word_lookup for si in s])]
# [len(x) for x in selected_sentences.values()],len(natural_sequences)
# [len(natural_sequences[i]) for i in natural_sequence_lens]

# doc.noun_phrases
# all_noun_phrases = []
# for sentence_len in selected_sentences:
#     for sentence in tqdm(selected_sentences[sentence_len]):
#         doc = nlp(" ".join(sentence))
#         all_noun_phrases.extend([chunk.text for chunk in doc.noun_chunks])
# noun_phrases = {i:[x.split() for x in all_noun_phrases if len(x.split()) == i] for i in range(4,9)}