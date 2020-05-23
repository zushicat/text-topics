# text-topics


NMF implementation of 2 cases:
- if you know the number of topics: nmf_fixed_k.py
- if you **don't** know the number of topics: nmf_unknown_k.py

LDA implementation incl. grid search for unknown number of topics (k): lda.py    
(This is for the sake of completeness, since the LDA results are not as good as those of NMF.)


### Data
The directories in /data:
- source_texts: Excerpts of wikipedia biographies falling in 3 broad topics:
    - Tudor dynasty (marked with "a")
    - Midcentury Architects / Designer (marked with "b")
    - Stars of the silent movie area (marked with "c")
- target_texts: Very short texts based on source texts whith varying similarity, marked accordingly to the source texts. Also, one text about a movie star not included in source texts and one text about "Charlie Brown" without any topic affiliation (marked with "d").

### Further Reading
#### General
- "Topic Analysis": https://monkeylearn.com/topic-analysis/
- "Latent Semantic Analysis using Python": https://www.datacamp.com/community/tutorials/discovering-hidden-topics-python
#### Text Cleaning (NLTK)
- "Stemming and Lemmatization in Python": datacamp.com/community/tutorials/stemming-lemmatization-python
#### Tokenizer / Vectorizer
- "An Intuitive Understanding of Word Embeddings: From Count Vectors to Word2Vec": https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/
- Scikit Learn documentation
    - "Count Vectorizer": https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    - "TFIDF Vectorizer": https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

#### LDA / NMF
- Scikit Learn documentation
    - "LDA": https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html
    - "NMF": https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
    - "NMF Example": https://scikit-learn.org/0.15/auto_examples/applications/topics_extraction_with_nmf.html
    - "Topic extraction with Non-negative Matrix Factorization and Latent Dirichlet Allocation": https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
- General
    - "Topic Modeling - Intro & Implementation": https://www.kaggle.com/akashram/topic-modeling-intro-implementation
    - "Topic Modelling with Scikit-learn": http://derekgreene.com/slides/topic-modelling-with-scikitlearn.pdf
    - "Using Machine Learning to Analyze Taylor Swift's Lyrics": https://news.codecademy.com/taylor-swift-lyrics-machine-learning/
    - "LDA in Python – How to grid search best topic models?": https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/
    - "Topic Modeling Quora Questions with LDA & NMF": https://towardsdatascience.com/topic-modeling-quora-questions-with-lda-nmf-aff8dce5e1dd
- Hyperparameter Tuning
    - "Topic Modeling using NMF and LDA using sklearn": https://shravan-kuchkula.github.io/topic-modeling/#gridsearch-the-best-lda-model
    - "LDA in Python – How to grid search best topic models?": https://www.machinelearningplus.com/nlp/topic-modeling-python-sklearn-examples/#11howtogridsearchthebestldamodel
- Topic Coherence (unknown number of topics k)
    - "Evaluation of Topic Modeling: Topic Coherence": https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
    - "Evaluate Topic Models: Latent Dirichlet Allocation (LDA)": https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
    - "derekgreene/topic-model-tutorial": https://github.com/derekgreene/topic-model-tutorial/blob/master/3%20-%20Parameter%20Selection%20for%20NMF.ipynb
    - "Topic modelling with NMF": https://nbviewer.jupyter.org/urls/gitlab8.trifork.nl/sofiah/topic-modelling-blog/raw/master/notebooks/topic-modelling-nmf.ipynb