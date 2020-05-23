from itertools import combinations
import json
from typing import Any, Dict, List, Set

# load external scripts / class
from load_data import get_texts
from vectorizer import Vectorizer

import gensim
import numpy as np
from sklearn.decomposition import NMF

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def _train_nmf(n_topics: int, vector_matrix: Any) -> Any:
    '''
    For further information about NMF hyperparameter, please refer:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    '''
    nmf = NMF(n_components=n_topics, init="nndsvd").fit(vector_matrix)
    return nmf


def _get_topics_top_words(nmf_model: Any, feature_names: List[str], n_top_words: int) -> List[List[str]]:
    '''
    This is a little different than the same named funktion in nmf_fixed_k.py: 
    returns list of lists with top words (instead of dict with strings)
    '''
    topics = []
    for _, topic in enumerate(nmf_model.components_):
        topics.append([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    return topics


def _get_topic_coherence(word_2_vec_lookup: Any, topics_words: List[List[str]]) -> float:
    topic_coherence: float = 0.0
    for topic_index in range(len(topics_words)):  # each topic (for current k)
        pair_scores: List[float] = []
        for pair in combinations(topics_words[topic_index], 2):  # each word pairs (2-word combinations) in current topic
            pair_scores.append(word_2_vec_lookup.similarity(pair[0], pair[1]))  # get vector similarity in word_2_vec
        
        topic_score: float = sum(pair_scores) / len(pair_scores)  # normalize
        topic_coherence += topic_score

    return topic_coherence / len(topics_words)  # normalize (mean score across all topics)


def _get_best_k(doc_texts: Set[str], vector_matrix: Any, kmin: int, kmax: int, n_top_words: int) -> int:
    # ***
    # create (gensim) word 2 vector model
    docs_tokenized: List[str] = [x.split() for x in doc_texts]
    word_2_vec_model = gensim.models.Word2Vec(docs_tokenized, size=1000, min_count=1, sg=1)
    word_2_vec_lookup = word_2_vec_model.wv

    # ***
    # For each number of topics (k):
    # get the best topic coherence of word_2_vec_lookup and top words in topics
    k_values = []
    topic_coherences: Dict[str, float] = {}
    for k in range(kmin, kmax+1):
        k_values.append(k)

        nmf_model = _train_nmf(k, vector_matrix)
        nmf_W = nmf_model.transform(vector_matrix)
        nmf_H = nmf_model.components_

        topics_words = _get_topics_top_words(nmf_model, feature_names, n_top_words)
        topic_coherences[k] = _get_topic_coherence(word_2_vec_lookup, topics_words)

    return max(topic_coherences, key=topic_coherences.get)  # return k with max. coherence value


if __name__ == '__main__':
    vectorizer = Vectorizer(True)  # create instance of vectorizer: True (or None): tfidf, False: Count Vectorizer

    source_texts: Dict[str, str] = get_texts("source_texts")  # get normalized/cleaned texts from passed directory
    vector_matrix, feature_names = vectorizer.create_vector_matrix_Tfidf(source_texts)  # get vector matrix and list of token (features)

    # ***
    # when number of topics is unknown: estimate best number of k
    kmin = 3   # choose sensible range for number of topics
    kmax = 5
    n_top_words = 10 

    best_k = _get_best_k(source_texts.values(), vector_matrix, kmin, kmax, n_top_words)


    # ***************************
    # following is the same as in nmf_fixed_k.py
    # ***************************

    # ***
    # train nmf with estimated number of topics
    n_topics = best_k
    nmf_model = _train_nmf(n_topics, vector_matrix)

    # ***
    # explore topics and create human readable topic name list; add attribute for unknown topics (see prediction below)
    # ! this hard coded list depends on input and is only fixed as long as nothing changes !
    topics_top_words: Dict[str, str] = _get_topics_top_words(nmf_model, feature_names, n_top_words)
    # print(json.dumps([" ".join(x) for x in topics_top_words], indent=2))
    topic_names: List[str] = ["bio_tudor", "bio_silent_movie_stars", "bio_design_arch", "unknown_topic"]
    

    # ***
    # check out source text topics
    titles_by_topics: Dict[str, List[str]] = {x:[] for x in topic_names}
    
    train_y: Any = nmf_model.transform(vector_matrix)
    for i, p in enumerate(train_y):
        doc_title = list(source_texts.keys())[i]  # get title (filename) assoc. with doc
        predicted_topic_index = np.argmax(p)  # get most relevant topic index
        topic_name = topic_names[predicted_topic_index]  # het human readable topic with index
        titles_by_topics[topic_name].append(doc_title)  # append document title
    
    # print(json.dumps(titles_by_topics, indent=2))

    # ***
    # predict topics of short test texts
    titles_by_topics: Dict[str, List[str]] = {x:[] for x in topic_names}  # re-init

    # there is also 1 text about Charlie Brown (wikipedia_d_1.txt) which belongs to the unknown topic "peanuts"
    # Hence, define a threshold for the topic destribution: below this value is unknwon
    threshold = 0.1
    
    target_texts = get_texts("target_texts")  # get short test texts
    pred_vector_matrix = vectorizer.transform_documents_to_vectormatrix(target_texts.values())  # get vector matrix of new texts with fitted vectorizer
    pred_topic_distribution = nmf_model.transform(pred_vector_matrix)  # get topic probabiliy
    for i, p in enumerate(pred_topic_distribution):  # same as above
        doc_title = list(target_texts.keys())[i]
        
        # print(doc_title, p)  # see topic distribution for each text
        
        predicted_topic_index = np.argmax(p)  # get index of max. value
        if p[predicted_topic_index] < threshold:
            topic_name = "unknown_topic"
        else:
            topic_name = topic_names[predicted_topic_index]
        titles_by_topics[topic_name].append(doc_title)
    
    print(json.dumps(titles_by_topics, indent=2))