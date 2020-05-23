import json
from typing import Any, Dict, List

# load external scripts / class
from load_data import get_texts
from vectorizer import Vectorizer

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def _train_lda(n_topics: int, vector_matrix: Any) -> Any:
    '''
    For further information about NMF hyperparameter, please refer:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    '''
    lda = LDA(n_components=n_topics).fit(vector_matrix)
    return lda


def _grid_search_lda(vector_matrix: Any) -> Any:
    '''
    For further information about LDA hyperparameter, please refer:
    https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    
    ---
    model.best_params_ from grid search of params. below:
    {'learning_decay': 1.0, 'learning_offset': 8, 'max_iter': 50, 'n_components': 3}
    '''
    search_params = {
        "n_components": [3, 4, 5],  # number of topics
        "max_iter": [10, 50, 100],
        "learning_offset": [6, 8, 10],
        "learning_decay": [0.5, 0.8, 1.0],
    }
    
    model = GridSearchCV(
        LDA(), 
        param_grid=search_params
    ).fit(vector_matrix)
    
    print(model.best_params_)  # plot best est. parameter from search_params

    return model.best_estimator_  # return model with best param.


def _get_topic_top_words(nmf: Any, n_top_words: int) -> Dict[str, str]:
    '''
    This is a little different than the same named funktion in nmf_unknown_k.py: 
    returns dict with topic index: string of top words (instead of list of lists with top words)
    '''
    topics = {}
    for topic_idx, topic in enumerate(nmf.components_):
        topics[f"topics_{topic_idx}"] = " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])

    return topics


if __name__ == '__main__':
    vectorizer = Vectorizer(False)  # create instance of vectorizer: True (or None): tfidf, False: Count Vectorizer

    source_texts: Dict[str, str] = get_texts("source_texts")  # get normalized/cleaned texts from passed directory
    vector_matrix, feature_names = vectorizer.create_vector_matrix_Tfidf(source_texts)  # get vector matrix and list of token (features)

    # ***
    # train lda
    # a) number of k is known from source material: 3 broad types of biographies
    # n_topics = 3 
    # lda_model = _train_lda(n_topics, vector_matrix)
    
    # OR b) get estimated number of topics from grid search (can take a little while, depending on num. param. in gridsearch)
    # lda_model = _grid_search_lda(vector_matrix)

    # let's just take the best parameters from grid search as a shortcut
    # but you should definitely play arounnd with the hyperparameters
    best_params: Dict[str, Any] = {'learning_decay': 1.0, 'learning_offset': 8, 'max_iter': 50, 'n_components': 3}
    lda_model = LDA(**best_params).fit(vector_matrix)

    # ***
    # explore topics and create human readable topic name list; add attribute for unknown topics (see prediction below)
    # ! this hard coded list depends on input and is only fixed as long as nothing changes !
    n_top_words = 15
    topics_top_words: Dict[str, str] = _get_topic_top_words(lda_model, n_top_words)

    print(json.dumps(topics_top_words, indent=2))
    topic_names: List[str] = ["bio_tudor", "bio_design_arch", "bio_silent_movie_stars", "unknown_topic"]

    # ***************************
    # following is the same as in nmf_fixed_k.py and nmf_unknown.k.py (except model name)
    # ***************************

    # ***
    # check out source text topics
    titles_by_topics: Dict[str, List[str]] = {x:[] for x in topic_names}
    
    train_y: Any = lda_model.transform(vector_matrix)
    for i, p in enumerate(train_y):
        doc_title = list(source_texts.keys())[i]  # get title (filename) assoc. with doc
        predicted_topic_index = np.argmax(p)  # get most relevant topic index
        topic_name = topic_names[predicted_topic_index]  # get human readable topic with index
        titles_by_topics[topic_name].append(doc_title)  # append document title
    
    print(json.dumps(titles_by_topics, indent=2))

    # ***
    # predict topics of short test texts
    titles_by_topics: Dict[str, List[str]] = {x:[] for x in topic_names}  # re-init
    
    # there is also 1 text about "Charlie Brown" (wikipedia_d_1.txt) which belongs to the unknown topic "peanuts"
    # Hence, define a threshold for the topic destribution: below this value text topic is unknwon
    threshold = 0.1
    
    target_texts = get_texts("target_texts")  # get short test texts
    pred_vector_matrix = vectorizer.transform_documents_to_vectormatrix(target_texts.values())  # get vector matrix of new texts with fitted vectorizer
    pred_topic_distribution = lda_model.transform(pred_vector_matrix)  # get topic probabiliy
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

    # Conclusion:
    # Especially compared to NMF, even with grid search the results on this train set are far from good.
