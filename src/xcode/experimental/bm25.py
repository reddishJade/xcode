from __future__ import annotations

import math
from typing import Sequence


class BM25Okapi:
    """无 numpy 依赖的轻量级 BM25Okapi 算法实现。"""

    def __init__(
        self, corpus: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = (
            sum(len(doc) for doc in corpus) / self.corpus_size
            if self.corpus_size > 0
            else 0.0
        )
        self.doc_lengths = [len(doc) for doc in corpus]

        self.doc_freqs: dict[str, int] = {}
        self.term_freqs: list[dict[str, int]] = []

        for doc in corpus:
            frequencies: dict[str, int] = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.term_freqs.append(frequencies)

            for word in frequencies:
                self.doc_freqs[word] = self.doc_freqs.get(word, 0) + 1

        self.idfs: dict[str, float] = {}
        for word, freq in self.doc_freqs.items():
            self.idfs[word] = math.log(
                (self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def get_scores(self, query: Sequence[str]) -> list[float]:
        """计算查询与语料库中所有文档的 BM25 得分。"""
        scores: list[float] = []
        if self.corpus_size == 0 or self.avgdl == 0.0:
            return [0.0] * self.corpus_size

        for doc_idx in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            frequencies = self.term_freqs[doc_idx]

            for word in query:
                if word not in frequencies:
                    continue
                idf = self.idfs.get(word, 0.0)
                if idf < 0:
                    idf = 0.0
                tf = frequencies[word]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * (doc_len / self.avgdl)
                )
                score += idf * (numerator / denominator)
            scores.append(score)
        return scores
