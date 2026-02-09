import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("punkt")

def compute_redundancy(summary):
    sentences = nltk.sent_tokenize(summary)

    if len(sentences) <= 1:
        return 0.0

    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sentences)
    sim = cosine_similarity(tfidf)

    upper = sim[np.triu_indices(len(sentences), k=1)]
    return upper.mean() if len(upper) > 0 else 0.0