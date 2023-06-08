import numpy as np


class TfIdf:
    def __init__(self, stop_words: set = None):
        self.stop_words = stop_words or set()
        self.vocabulary_ = dict()
        self.doc_voc_: dict[int, dict] = dict()

    def fit_transform(self, raw_documents: list[str]) -> np.ndarray:
        for i, doc in enumerate(raw_documents):
            self.doc_voc_[i] = self.doc_voc_.get(i, dict())

            for word in doc.split():
                if word in self.stop_words:
                    continue
                self.doc_voc_[i][word] = self.doc_voc_[i].get(word, 0) + 1
                self.vocabulary_[word] = self.vocabulary_.get(word, 0) + 1

        return self.transform(raw_documents)

    def transform(self, raw_documents: list[str]) -> np.ndarray:
        mx = np.zeros(shape=(len(raw_documents), len(self.vocabulary_.keys())))
        words = sorted(self.vocabulary_.keys())

        for i, doc in enumerate(raw_documents):
            for word in doc.split():
                if word in self.stop_words or word not in words:
                    continue
                mx[i, words.index(word)] += 1

        return self.normalize(mx)
    
    def normalize(self, mx: np.ndarray) -> np.ndarray:
        # частота слова в тексте
        tf = np.divide(
            mx,
            np.sum(mx, axis=1, keepdims=True),
            out=np.zeros_like(mx, dtype=float),
            where=np.sum(mx, axis=1, keepdims=True) != 0
        )
        
        # log((общее число документов + 1) / (число документов, содержащих данное слово + 1)) + 1
        idf = np.log((mx.shape[0] + 1) / (np.sum(mx != 0, axis=0) + 1)) + 1

        return tf * idf


# mx = np.array([
#     [2, 0, 1, 1],
#     [1, 1, 0, 0]
# ])
# print(TfIdf().normalize(mx))
