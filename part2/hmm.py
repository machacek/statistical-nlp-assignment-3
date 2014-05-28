#!/usr/bin/env python3
from collection import namedtuple, Counter
from math import log2
from itertools import zip_longest

class SupervisedHMMTagger(object):
    def __init__(self, train_data, n=3):
        self.n = n 
        self.lambdas = (n+1) * (1,) # pocatecni parametry vylazovani, pozor, parametry jsou v obracenem poradi
        self.c_tw = Counter() # pocet videnycn dvojic tag, slovo (c_wt ze slidu)
        self.c_t  = Counter() # pocet videnych tagu (c_t ze slidu)
        self.c_h  = Counter() # pocet videnych historii tagu (c_t(n-1) ze slidu)
        self.c_ht = Counter() # pocet videnych tag n-gramu (c_tn ze slidu) 

    def train_file(self, file):
        for line in file:
            sentence = [item.split('/',2) for item in line.split()]
            self.train_sentence(sentence)

    def train_sentence(self, sentence):
        sentence = list(sentence)
        words = [word for word, tag in sentence]
        tags = [tag for word, tag in sentence]
        for word, tag, tag_history in zip(words, tags, self.history_generator(tags)):
            self.c_tw[tag,word] += 1
            self.c_t[tag] += 1
            for suffix in suffixes(tag_history):
                self.c_h[suffix] += 1
                self.c_ht[suffix, tag] += 1

    def train_lambdas(self, held_out_data):
        tags = [tag for word,tag in held_out_data]
        epsilon = 0.001

        # Debug output
        print("Starting Smoothing EM Algorithm", file=sys.stderr)
        print("Actual Lambdas:", self.lambdas, file=sys.stderr)
        print("Actual Cross Entropy:", self.cross_entropy(tags), file=sys.stderr)

        done = False
        iteration = 0
        while not done:
            iteration += 1
            print("\nStarting iteration %s" % iteration, file=sys.stderr)

            # Prepare list of lambda multiplicators
            lambdas = [0 for _ in self.lambdas]

            for tag, tag_history in zip(tags, history_generator(tags)):
                interpolated_prob = self.tag_probability(tag, tag_history)
                for i, suffix in enumerate(suffixes(tag_history)):
                    lambdas[i] += self.n_tag_probability(tag, suffix) / interpolated_prob
                lambdas[-1] +=  1 / (self.vocabulary_size() * interpolated_prob)
            
            # Multiply with old lambdas and normalize
            lambdas = [lambda_ * lambda_mul for lambda_, lambda_mul in zip(self.lambdas, lambdas)]
            sum_ = sum(lambdas)
            lambdas = [lambda_ / sum_ for lambda_ in lambdas]

            # Check if some parameter change significantly and continue in next iteration
            done = True
            for old_lambda, new_lambda in zip(self.lambdas, lambdas):
                if abs(old_lambda - new_lambda) > epsilon:
                    done = False

            # Apply new Lambdas
            self.lambdas = lambdas
            
            print("New Lambdas:", self.lambdas, file=sys.stderr)
            print("New Cross Entropy:", self.tag_cross_entropy(tags), file=sys.stderr)

        print("End of EM Smoothing Algorithm", file=sys.stderr)

    def tag_cross_entropy(self, held_out_tags):
        sum = 0
        tags = [tag for word,tag in held_out_tags]
        for tag, tag_history in zip(tags, history_generator(tags)):
            sum += self.log_tag_probability(tag, tag_history)
        return -sum / len(tags);

    def decode_sentence(self, sentence):
        words = list(sentence)

        # Viterbi algorithm
        TrelisNode = namedtuple("TrelisNode", ["log_alpha","previous_node","tag"])
        stage = { self.start_state() : TrelisNode(log_alpha=0, previous=None, tag=None) }
        for word in words:
            new_stage = {}
            for previous_state, previous_node in stage.items():
                for tag in self.possible_tags(word):

                    log_alpha = previous_node.log_alpha \
                            + self.log_tag_probability(tag, previous_state) \
                            + self.log_word_probability(word, tag)

                    state = previous_state[1:] + (tag,)

                    if state not in new_stage_indexed or new_stage_indexed[state].log_alpha < log_alpha:
                        new_node = TrelisNode(log_alpha=log_alpha, previous=previous_node, tag=tag)
                        new_stage_indexed[state] = new_node

            stage = new_stage

        # Backtrace the best tag path
        node = max(stage.values(), key=lambda x: x.log_alpha)
        rev_tags = []
        while node.tag is not None:
            rev_tags.append(node.tag)
            node = node.previous_node
        return reversed(rev_tags)

    def possible_tags(self, word):
        pass

    def vocabulary_size(self, tag=None):
        pass

    def tag_probability(self, tag, tag_history):
        sum = 0
        for lambda_coeff, suffix in safe_zip(self.lambdas[:-1], suffixes(tag_history)):
            sum += lambda_coeff * self.n_tag_probability(tag, suffix)
        sum += self.lambdas[-1] / self.vocabulary_size()
        return sum

    def n_tag_probability(self, tag, tag_history):
        try:
            return self.c_ht[suffix, tag] / self.c_h[suffix] 
        except ZeroDivisionError:
            return 1/self.vocabulary_size()

    def word_probability(self, word, tag):
        nominator = self.c_tw[tag, word] + 1
        denominator = self.c_t[tag] + self.vocabulary_size(tag)
        return nominator / denominator

    def log_tag_probability(self, tag, tag_history):
        return log2(self.tag_probability(tag, tag_history))

    def log_word_probability(self, word, tag):
        return log2(self.log_word_probability(word, tag))

    def start_state(self):
        return (self.n - 1) * (None,)

    def history_generator(iterable):
        iterator = iter(iterable)
        n_gram = deque(self.start_state())
        while True:
            new_item = next(iterator)
            yield tuple(n_gram)
            n_gram.popleft()
            n_gram.append(new_item)

def safe_zip(*args):
    for n_tuple in zip_longest(*args):
        assert None not in n_tuple
        yield n_tuple

def suffixes(sequence):
    for i in range(len(sequence) + 1):
        yield sequence[i:]



