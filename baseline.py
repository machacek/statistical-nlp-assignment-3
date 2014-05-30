#!/usr/bin/env python3
from collections import Counter, defaultdict

class BaselineTagger(object):
    def __init__(self):
        self.lexicon = defaultdict(Counter)
        self.all_tags = Counter()

    def train_parameters(self, train_data):
        for sentence in train_data:
            for word, tag in sentence:
                self.lexicon[word][tag] += 1
                self.all_tags[tag] += 1
        self.finalize()

    def finalize(self):
        self.words_tags = {}
        for word, tag_counter in self.lexicon.items():
            self.words_tags[word] = tag_counter.most_common(1)[0][0]
        del self.lexicon

        self.unknown_word_tag = self.all_tags.most_common(1)[0][0]
        del self.all_tags

    def decode(self, data):
        for sentence in data:
            yield list(self.decode_sentence(sentence))

    def decode_sentence(self, sentence):
        for word in sentence:
            try:
                yield word, self.words_tags[word]
            except KeyError:
                yield word, self.unknown_word_tag

