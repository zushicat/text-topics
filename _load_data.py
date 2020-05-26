import os
from typing import Dict, List

from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


STOPWORDS = set(stopwords.words('english'))
TOKENIZER = RegexpTokenizer(r"\w+")  # filter punctuation etc. / all except [a-zA-Z0-9_]+
LEMMATIZER = WordNetLemmatizer()


def get_texts(dir_name: str) -> Dict[str, str]:
    '''
    Get texts from passed directory and return cleaned texts, separated in paragraphs
    '''
    def remove_stopwords(token: List[str]) -> List[str]:
        return [w for w in token if not w in STOPWORDS]  # filtered token
    def tokenize_text(text: str) -> List[str]:
        return TOKENIZER.tokenize(text)  # list of token without punctuation etc.
    def get_token_lemmata(token: List[str]) -> List[str]:
        return [LEMMATIZER.lemmatize(t) for t in token]

    file_names = os.listdir(f"data/{dir_name}")
    texts: Dict[str, List[str]] = {}
    for file_name in file_names:
        if file_name.find(".txt") == -1:
            continue

        text_name = file_name.replace(".txt", "")
        with open(f"data/{dir_name}/{file_name}") as f:
           texts[text_name] = f.read().replace("\n\n", " ")  # remove paragraphs

        # ***
        # simplify text in paragraphs
        document_token = remove_stopwords(tokenize_text(texts[text_name].lower()))
        document_token_lemmatized = get_token_lemmata(document_token)  # prefer lemmatization over stemming
        texts[text_name] = " ".join(document_token_lemmatized)
        
    return texts


def get_texts_simple(dir_name: str) -> Dict[str, str]:
    '''
    Lowercase text and remove punctuation only
    '''
    def tokenize_text(text: str) -> List[str]:
        return TOKENIZER.tokenize(text)  # list of token without punctuation etc.
    
    file_names = os.listdir(f"data/{dir_name}")
    texts: Dict[str, List[str]] = {}
    for file_name in file_names:
        if file_name.find(".txt") == -1:
            continue

        text_name = file_name.replace(".txt", "")
        with open(f"data/{dir_name}/{file_name}") as f:
           texts[text_name] = f.read().replace("\n\n", " ")  # remove paragraphs

        document_token = tokenize_text(texts[text_name])
        texts[text_name] = " ".join(document_token)
        
    return texts
