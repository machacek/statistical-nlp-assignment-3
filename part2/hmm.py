#!/usr/bin/env python3
from collection import namedtuple

class SupervisedHMMTagger(object):
    def __init__(self, train_data, n=3):
        self.n = n

    def train_sentence(self, sentence):
        sentence = list(sentence)
        words = [word for word, tag in sentence]
        tags = [tag for word, tag in sentence]
        for word, tag, tag_n_gram in zip(words, tags, n_gram_generator):
            pass


    def tag(self, data_to_tag):
        pass

    def viterbi(self, data_to_tag):

        TrelisNode = namedtuple("TrelisNode", ["state","probability","previous"])

        stage = [TrelisNode(state=self.get_start_state(), probability=1, previous=None)]
        for word in data_to_tag:
            new_stage = []
            
            for node in stage:
           

    def get_start_state(self):
        return (None,) * self.get_n()

    def get_n(self):
        return self.n

def n_grams(iterable, n=2):
    iterator = iter(iterable)
    actual_n_gram = deque(None for _ in range(n))

    # fill the first n items to deque
    for _ in range(n):
        actual_n_gram.append(next(iterator))

    while True:
        actual_n_gram.popleft()
        actual_n_gram.append(next(iterator))
        yield tuple(actual_n_gram)
