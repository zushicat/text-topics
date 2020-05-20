import json
import os
from typing import Any, Dict, List, Tuple

from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

STOPWORDS = set(stopwords.words('english'))
TOKENIZER = RegexpTokenizer(r"\w+")  # filter punctuation etc. / all except [a-zA-Z0-9_]+
LEMMATIZER = WordNetLemmatizer()
VECTORIZER = TfidfVectorizer()


def _get_texts(dir_name: str) -> Dict[str, str]:
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


def _get_vector_matrix_Tfidf(texts: Dict[str, str]) -> Tuple[Any, List[str]]:
    corpus: List[str] = list(texts.values())  # each list element is a document
    
    vector_matrix = VECTORIZER.fit_transform(corpus)  # returns sparse matrix, [n_samples, n_features]
    feature_names = VECTORIZER.get_feature_names()  # returns list of token (feature names)
    
    return vector_matrix, feature_names


def _train_nmf(n_topics: int, vector_matrix: Any) -> Any:
    nmf = NMF(n_components=n_topics, init="nndsvd").fit(vector_matrix)
    return nmf


def _get_topic_top_words(nmf: Any, n_top_words: int) -> Dict[str, str]:
        topics = {}
        for topic_idx, topic in enumerate(nmf.components_):
            topics[f"topics_{topic_idx}"] = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

        return topics


if __name__ == '__main__':
    source_texts = _get_texts("source_texts")
    vector_matrix, feature_names = _get_vector_matrix_Tfidf(source_texts)

    n_topics = 3  # this is known from source material: 3 broad kinds of biographies
    nmf_model = _train_nmf(n_topics, vector_matrix)

    # ***
    # explore topics and create human readable topic name list
    # ! this hard coded list depends on input and is only fixed as long as nothing changes !
    n_top_words = 15
    topics_top_words = _get_topic_top_words(nmf_model, n_top_words)

    # print(json.dumps(topics_top_words, indent=2))
    topic_names = ["bio_tudor", "bio_silent_movie_stars", "bio_design_arch"]

    # ***
    # check out source text topics
    titles_by_topics: Dict[str, List[str]] = {x:[] for x in topic_names}
    
    train_y = nmf_model.transform(vector_matrix)
    for i, p in enumerate(train_y):
        doc_title = list(source_texts.keys())[i]  # get title (filename) assoc. with doc
        predicted_topic_index = np.argmax(p)  # get most relevant topic index
        topic_name = topic_names[predicted_topic_index]  # het human readable topic with index
        titles_by_topics[topic_name].append(doc_title)  # append document title
    
    # print(json.dumps(titles_by_topics, indent=2))

    # ***
    # predict topics of short test texts
    titles_by_topics: Dict[str, List[str]] = {x:[] for x in topic_names}  # re-init
    
    target_texts = _get_texts("target_texts")  # get short test texts
    pred_vector_matrix = VECTORIZER.transform(target_texts.values())  # get vector matrix of new texts with fitted vectorizer
    pred_y = nmf_model.transform(pred_vector_matrix)  # get topic probabiliy
    for i, p in enumerate(pred_y):  # same as above
        doc_title = list(target_texts.keys())[i]
        predicted_topic_index = np.argmax(p)
        topic_name = topic_names[predicted_topic_index]
        titles_by_topics[topic_name].append(doc_title)
    
    print(json.dumps(titles_by_topics, indent=2))
