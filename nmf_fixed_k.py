import json
from typing import Any, Dict, List

# load external scripts / class
from load_data import get_texts
from vectorizer import Vectorizer

import numpy as np
from sklearn.decomposition import NMF

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def _train_nmf(n_topics: int, vector_matrix: Any) -> Any:
    nmf = NMF(n_components=n_topics, init="nndsvd").fit(vector_matrix)
    return nmf


def _get_topic_top_words(nmf: Any, n_top_words: int) -> Dict[str, str]:
        topics = {}
        for topic_idx, topic in enumerate(nmf.components_):
            topics[f"topics_{topic_idx}"] = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

        return topics


if __name__ == '__main__':
    vectorizer = Vectorizer(True)  # create instance of vectorizer: True (or None): tfidf, False: Count Vectorizer

    source_texts: Dict[str, str] = get_texts("source_texts")  # get normalized/cleaned texts from passed directory
    vector_matrix, feature_names = vectorizer.create_vector_matrix_Tfidf(source_texts)  # get vector matrix and list of token (features)

    # ***
    # train nmf
    # number of k is known from source material: 3 broad types of biographies
    n_topics = 3 
    nmf_model = _train_nmf(n_topics, vector_matrix)

    # ***
    # explore topics and create human readable topic name list
    # ! this hard coded list depends on input and is only fixed as long as nothing changes !
    n_top_words = 15
    topics_top_words: Dict[str, str] = _get_topic_top_words(nmf_model, n_top_words)

    # print(json.dumps(topics_top_words, indent=2))
    topic_names: List[str] = ["bio_tudor", "bio_silent_movie_stars", "bio_design_arch"]

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
    
    target_texts = get_texts("target_texts")  # get short test texts
    pred_vector_matrix = vectorizer.transform_documents_to_vectormatrix(target_texts.values())  # get vector matrix of new texts with fitted vectorizer
    pred_y = nmf_model.transform(pred_vector_matrix)  # get topic probabiliy
    for i, p in enumerate(pred_y):  # same as above
        doc_title = list(target_texts.keys())[i]
        predicted_topic_index = np.argmax(p)
        topic_name = topic_names[predicted_topic_index]
        titles_by_topics[topic_name].append(doc_title)
    
    print(json.dumps(titles_by_topics, indent=2))
