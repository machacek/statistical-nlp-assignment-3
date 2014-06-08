#!/usr/bin/env python3
from collections import Counter, defaultdict

class BaselineTagger(object):
    def __init__(self):
        self.lexicon = defaultdict(Counter)
        self.all_tags = Counter()

    def train_labeled(self, labeled_data):
        for word, tag in labeled_data:
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
        for word in data:
            try:
                yield word, self.words_tags[word]
            except KeyError:
                yield word, self.unknown_word_tag

