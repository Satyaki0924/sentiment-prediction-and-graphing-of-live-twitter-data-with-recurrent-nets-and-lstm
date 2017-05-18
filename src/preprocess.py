from __future__ import print_function
from string import punctuation
from collections import Counter
import pickle
import os
import numpy as np


class Process(object):
    def __init__(self):
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.reviews = None
        self.labels = None
        self.words = None
        self.features = None

        if os.path.exists(os.path.dirname(os.path.realpath(self.path)) + '/vocab_list/data_vocab.p'):
            self.shouldGo = False
            with open(os.path.dirname(os.path.realpath(self.path)) + '/vocab_list/data_vocab.p', 'rb') as f:
                self.vocab_to_int = pickle.load(f)
        else:
            self.vocab_to_int = None
            self.shouldGo = True
        self.review_int = list()
        self.non_zero_idx = []

    def process_reviews(self, reviews):
        print('processing review...')
        self.reviews = reviews
        all_text = ''.join([c for c in self.reviews if c not in punctuation])
        self.reviews = all_text.split('\n')
        all_text = ' '.join(self.reviews)
        self.words = all_text.split()
        if self.shouldGo:
            self.encode_vocab()
        self.encode_review()
        self.remove_zero_vectors('r')
        self.feature_matrix()
        return self.features, self.non_zero_idx

    def encode_vocab(self):
        counts = Counter(self.words)
        vocab = sorted(counts, key=counts.get, reverse=True)
        self.vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}
        with open(os.path.dirname(os.path.realpath(self.path)) + '/vocab_list/data_vocab.p', 'wb') as f:
            pickle.dump(self.vocab_to_int, f)

    def encode_review(self):
        for review in self.reviews:
            self.review_int.append([self.vocab_to_int[word] for word in review.split() if word in self.vocab_to_int])

    def process_labels(self, labels, non_zero_idx):
        print('processing labels...')
        self.labels = labels.split('\n')
        self.labels = np.array([1 if label == 'positive' else 0 for label in self.labels])
        self.non_zero_idx = non_zero_idx
        self.remove_zero_vectors('l')
        return self.labels

    def remove_zero_vectors(self, q):
        if not self.non_zero_idx:
            self.non_zero_idx = [i for i, review in enumerate(self.review_int) if len(review) != 0]
        if q == 'r':
            self.review_int = [self.review_int[i] for i in self.non_zero_idx]
        elif q == 'l':
            self.labels = np.array([self.labels[i] for i in self.non_zero_idx])

    def feature_matrix(self):
        seq_length = 200
        self.features = np.zeros((len(self.review_int), seq_length), dtype=int)
        for i, j in enumerate(self.review_int):
            self.features[i, -len(j):] = np.array(j)[:seq_length]
