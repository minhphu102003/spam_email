"""Utility helpers for TF-IDF-based exploration."""
from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def top_tfidf_terms(
    texts: Iterable[str],
    *,
    n: int = 20,
    analyzer: str = "word",
    ngram_range=(1, 1),
    stop_words=None,
    lowercase: bool = True,
    max_features=None,
) -> List[Tuple[float, str]]:
    """Compute the top-n terms by mean TF-IDF weight across the provided texts."""
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        stop_words=stop_words,
        lowercase=lowercase,
        max_features=max_features,
    )
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()  # average TF-IDF per term
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(scores, terms), reverse=True)
    return ranked[:n]
