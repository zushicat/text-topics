# from collections import Counter
from itertools import chain, combinations
import json
from typing import Any, Dict, List, Tuple

from _load_data import get_texts_simple
from _request_wikipedia import request_wikipedia

from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


TOKENIZER = RegexpTokenizer(r"\w+")
LEMMATIZER = WordNetLemmatizer()
STOPWORDS = list(stopwords.words())


def _get_best_k(vector_matrix: Any, feature_list: List[str]) -> int:
        # ***
        # create (gensim) word 2 vector model
        word_2_vec_model = Word2Vec(docs_phrased, size=1000, min_count=1, sg=1)
        word_2_vec_lookup = word_2_vec_model.wv

        # ***
        # For each number of topics (k):
        # get the best topic coherence of word_2_vec_lookup and top words in topics
        kmin = 3
        kmax = 5
        n_top_words = 20  # number of topwords is debatable
        
        best_k = -1
        max_coherence = 0.0
        for k in range(kmin, kmax+1):
            nmf_model =  NMF(n_components=k, init="nndsvd").fit(vector_matrix)
            
            # ***
            # get top words of each topic to calculate coherence
            topics_words = _get_topics_top_words(nmf_model, feature_list, k)
            
            current_coherence = _get_topic_coherence(word_2_vec_lookup, topics_words)
            if current_coherence > max_coherence:
                max_coherence = current_coherence
                best_k = k
                
        return best_k


def _get_topic_coherence(word_2_vec_lookup: Any, topics_words: List[List[str]]) -> float:
    topic_coherence: float = 0.0
    for topic_index in range(len(topics_words)):  # each topic (for current k)
        pair_scores: List[float] = []
        for pair in combinations(topics_words[topic_index], 2):  # each word pairs (2-word combinations) in current topic
            pair_scores.append(word_2_vec_lookup.similarity(pair[0], pair[1]))  # get vector similarity in word_2_vec
        
        topic_score: float = sum(pair_scores) / len(pair_scores)  # normalize
        topic_coherence += topic_score

    return topic_coherence / len(topics_words)  # normalize (mean score across all topics)


def _get_topics_top_words(nmf_model: Any, feature_names: List[str], n_top_words: int) -> List[List[str]]:
    topics: List[List[str]] = []
    for _, topic in enumerate(nmf_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    return topics


def _filter_nouns(doc: str) -> List[str]:
    tokenized = TOKENIZER.tokenize(doc)
    nouns = [word for (word, pos) in pos_tag(tokenized) if(pos[:2] == 'NN')]
    return nouns


def _filter_stopwords(doc: str) -> List[str]:
    token_list = doc.split()
    filtered = [
        x for x in token_list 
            if x not in STOPWORDS and x.lower() not in ["articles", "references"]]  # plus wikipedia specific filter
    return filtered


def _get_lemmata(token: List[str]) -> List[str]:
    return [LEMMATIZER.lemmatize(t) for t in token]
    

if __name__ == '__main__':
    source_texts: Dict[str, str] = get_texts_simple("source_texts")  # get normalized text as dict <filename>:<doc text>
    
    # ******************************************************
    #
    # text preprocessing
    #
    # ******************************************************
    # ***
    # get (almost) unmodified text for Phrases training
    docs_token: List[List[str]] = [_get_lemmata(doc.lower().split()) for doc in source_texts.values()]  # lemmatize (almost) unmodified text
    
    # ***
    # get filtered text for further processing
    docs_filtered: List[List[str]] = [_filter_nouns(doc.lower()) for doc in source_texts.values()]  # get nouns only
    docs_filtered: List[List[str]] = [_get_lemmata(doc) for doc in docs_filtered]  # lemmatize filtered nouns
    
    # ***
    # get interrelated token in docs (bigrams) with gensom Phrases trained on (almost) unmodified texts
    # combined with "_" into 1 token i.e. "henry viii" -> "henry_viii"
    phrases_model = Phrases(docs_token, min_count=1, threshold=1)
    phrases = Phraser(phrases_model)
    
    docs_phrased = []  # <----- these are the docs / token for further processing: List of token per doc in List
    for doc in docs_filtered:
        docs_phrased.append(phrases[doc])

    
    # ******************************************************
    #
    # get top words of docs per topic
    #
    # ******************************************************
    # ***
    # create vector matrix (from texts) and vocabulary (features)
    docs_texts_str = [" ".join(x) for x in docs_phrased]  # pass List[str] to vectorizer
    
    # ***
    # get vector matrix and feature list
    vectorizer = TfidfVectorizer()
    vector_matrix = vectorizer.fit_transform(docs_texts_str)  
    feature_list: List[str] = vectorizer.get_feature_names()

    # ***
    # if you do not know number of topics (k):
    # k = _get_best_k(vector_matrix, feature_list)
    # otherwise: set k
    k = 3
    
    nmf_model = NMF(n_components=k, init="nndsvd").fit(vector_matrix)

    # ***
    # collect top phrases per doc under most relevant topic
    topic_docs: Dict[str, List[str]] = {}  # collect topwords of doc under topic
    pred_topic_distribution: Any = nmf_model.transform(vector_matrix)
    for i, dist in enumerate(pred_topic_distribution):
        # ***
        # get most relevant topic index
        predicted_topic_index = str(np.argmax(dist))  # get most relevant topic index
        
        # ***
        # get top words of document
        num_top_words = 5
        top_words = []
    
        doc_token_counted = zip(feature_list, np.asarray(vector_matrix[i].sum(axis=0)).ravel())  # ('1920s', 0), ...
        sorted_by_counts = sorted(doc_token_counted, key=lambda x: x[1], reverse=True)[:num_top_words]
        top_words = [x[0] for x in sorted_by_counts]
        
        # ***
        # collect top words of doc under topic
        if topic_docs.get(predicted_topic_index) is None:
           topic_docs[predicted_topic_index] = []
        topic_docs[predicted_topic_index].append(top_words)
    
    # ***
    # sort keys in dictionary
    topic_docs = {k: topic_docs[k] for k in sorted(topic_docs)}
    
    # ******************************************************
    #
    # request wikipedia categories per doc and get most relecant category per topic
    #
    # ******************************************************
    '''
    topic_categories: Dict[str, List[str]] = {}

    # ***
    # request wikipedia
    for topic_idx, top_word_lists in topic_docs.items():
        for top_word_list in top_word_lists:
            categories = request_wikipedia(top_word_list)

            if topic_categories.get(topic_idx) is None:
                topic_categories[topic_idx] = []
            
            for cat in categories:
                topic_categories[topic_idx].append(cat)
    
    # ***
    # if you like to gain some insight about collected categories: count categories per topic
    # for topic_idx, category_list in topic_categories.items():
    #     counted_categories = Counter(category_list)
    #     print(f"---- {topic_idx} ----")
    #     print(json.dumps(counted_categories, indent=2))
    '''

    # ***
    # let's assume the request is done and use this dummy result:
    with open("data/dummy_wikipedia_result.json", "r") as read_file:
        topic_categories: Dict[str, List[str]] = json.load(read_file)
    

    # ******************************************************
    #
    # process category phrases per topic
    #
    # ******************************************************
    # ***
    # pre-process the collected categories a little bit and create a (chainef) list of token
    cat_docs_token: List[List[str]] = [
        list(chain.from_iterable([_filter_stopwords(cat_str.lower().replace("-", " ")) for cat_str in doc]))
            for doc in topic_categories.values()
    ]
    
    # ***
    # get phrases from category docs
    phrases_model = Phrases(cat_docs_token, min_count=5, threshold=5)  # little stricter parameter
    phrases = Phraser(phrases_model)
    
    docs_phrased = []
    for doc in cat_docs_token:
        docs_phrased.append(phrases[doc])
    
    docs_phrased_str = [" ".join(doc) for doc in docs_phrased]

    # ***
    # vectorize
    vectorizer = CountVectorizer()
    vector_matrix = vectorizer.fit_transform(docs_phrased_str)  
    feature_list: List[str] = vectorizer.get_feature_names()

    # ***
    # get top 3 of the most representing phrases per topic
    num_top_words = 3
    for i, doc_matrix in enumerate(vector_matrix):
        doc_token_counted = zip(feature_list, np.asarray(doc_matrix.sum(axis=0)).ravel())
        sorted_by_counts = sorted(doc_token_counted, key=lambda x: x[1], reverse=True)[:num_top_words]
        top_words = [x[0].replace("_", " ") for x in sorted_by_counts]

        # ***
        # these are the top 3 phrases describing topic i
        print(f"topic {i}: {' | '.join(top_words)}")
