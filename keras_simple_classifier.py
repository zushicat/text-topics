import json
import os
from typing import Any, Dict, List, Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import gensim
import numpy as np

#import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

def _load_train_texts(dir_name: str) -> Tuple[List[str], List[str]]:
    # these str parts in filenames (i.e. wikipedia_a_1.txt) mark a topic of text
    topic_marker = {
        "_a_": "bio_tudor", 
        "_b_": "bio_design_arch", 
        "_c_": "bio_silent_movie_stars"
    }
    
    file_names = os.listdir(f"data/{dir_name}")
    texts: List[str] = []
    topics: List[str] = []

    for file_name in file_names:
        if file_name.find(".txt") == -1:
            continue

        try:
            # get topic from filename
            topic_found = False
            for i, t in enumerate(topic_marker.keys()):
                if file_name.find(t) != -1:
                    topics.append(list(topic_marker.values())[i])
                    topic_found = True
                    break
            if topic_found is False:  # ignore i.e. target_texts/wikipedia_d_1.txt
                continue

            # get assoc. text
            with open(f"data/{dir_name}/{file_name}") as f:
                text = f.read().replace("\n\n", " ")  # remove paragraphs
                texts.append(text)
        except Exception:
            continue
    
    return texts, topics


if __name__ == '__main__':
    # ***************************************************************
    #
    # text preprocessing
    #
    # ***************************************************************
    # ***
    # Get train texts and associated topics (of train texts): List[str], List[str]
    train_texts, train_topics = _load_train_texts("source_texts")
    
    # ***
    # preprocess texts and topics
    category_tags = list(set(train_topics))  # unique list of topics
    num_categories = len(category_tags)

    # ***
    # strings of token per document -> unique topicc list & index of topic per doc
    label_list = []
    for topic_token in train_topics:  # topic token per doc -> index in category_tags per doc
        label_list.append(category_tags.index(topic_token))  # append index in list

    tokenizer = Tokenizer() 
    tokenizer.fit_on_texts(train_texts)

    vocab_size = len(tokenizer.word_index)+1
    sequences = tokenizer.texts_to_sequences(train_texts)
    max_sequence_len =  len(max(sequences, key=len))

    train_X = pad_sequences(sequences)  # padded_sequences
    train_y = to_categorical(label_list)  # labels
    
    
    # ***************************************************************
    #
    # model
    #
    # ***************************************************************
    num_epochs = 100
    
    # ***
    # build model
    model = Sequential()
    
    # ***
    # simple model with embedding layer
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=50, input_length=max_sequence_len))
    model.add(layers.Dropout(0.15))  # Add dropout layer to prevent overfitting
    model.add(layers.GlobalMaxPool1D())  # down-sample input representation
    model.add(layers.Dense(num_categories, activation='sigmoid'))  # output layer

    model.summary()

    # ***
    # compile model (meaning: define matrix operations ("behavior") on model structure)
    model.compile(
        loss='categorical_crossentropy',  # multiclass
        optimizer='adam',
        metrics=['acc'])
    
    # ***
    # train the model on values (train_X) and labels (train_y)
    history = model.fit(
        train_X, 
        train_y, 
        epochs=num_epochs, 
        verbose=1,  # use verbose= 1 or 2 for output on training
    )  


    # ***************************************************************
    #
    # prediction
    #
    # ***************************************************************
    # ***
    # get text
    test_texts, test_topics = _load_train_texts("target_texts")

    # ***
    # preproccess
    test_label_list = []
    for topic_token in test_topics:  # topic token per doc -> index in category_tags per doc
        test_label_list.append(category_tags.index(topic_token))  # append index in list

    sequences = tokenizer.texts_to_sequences(test_texts)
    decoded_sentences = tokenizer.sequences_to_texts(sequences)

    test_X = pad_sequences(sequences, maxlen=max_sequence_len)  # padded_sequences
    test_y = to_categorical(test_label_list)  # test labels

    # ***
    # make prediction
    predictions = model.predict(test_X)

    for i, prediction in enumerate(predictions):
        category = category_tags[np.argmax(prediction)]
        print(f"----- document {i} -----")
        print(f"decoded text: {decoded_sentences[i][:50]}")
        print(f"prediction: {category}  | GT: {test_topics[i]}")