from typing import Any, Dict, List, Tuple

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class Vectorizer:
    def __init__(self, is_tfidf: bool = True):
        if is_tfidf is True:
            self.vectorizer = TfidfVectorizer()
        else:
            self.vectorizer = CountVectorizer()
  
    def create_vector_matrix(self, texts: Dict[str, str]) -> Tuple[Any, List[str]]:
        corpus: List[str] = list(texts.values())  # each list element is a document

        vector_matrix = self.vectorizer.fit_transform(corpus)  # returns sparse matrix, [n_samples, n_features]
        feature_names = self.vectorizer.get_feature_names()  # returns list of token (feature names)

        return vector_matrix, feature_names

    def transform_documents_to_vectormatrix(self, documents: List[str]) -> Any:
        return self.vectorizer.transform(documents)
