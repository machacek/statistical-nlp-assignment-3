#!/usr/bin/env python3
from collections import namedtuple, Counter, defaultdict, deque
from math import log2
from itertools import zip_longest
import sys
import heapq

class HMMTagger(object):
    def __init__(self, n=3):
        self.n = n 
        self.lambdas = (n+1) * (1/(n+1),) # pocatecni parametry vylazovani, pozor, parametry jsou v obracenem poradi
        self.c_tw = Counter() # pocet videnycn dvojic tag, slovo (c_wt ze slidu)
        self.c_t  = Counter() # pocet videnych tagu (c_t ze slidu)
        self.c_h  = Counter() # pocet videnych historii tagu (c_t(n-1) ze slidu)
        self.c_ht = Counter() # pocet videnych tag n-gramu (c_tn ze slidu) 
        self.word_lexicon = defaultdict(set)
        self.tag_lexicon = defaultdict(set)

    def train_parameters(self, train_data):
        for sentence in train_data:
            self.train_parameters_sentence(sentence)

    def train_parameters_sentence(self, sentence):
        words = [word for word, tag in sentence]
        tags = [tag for word, tag in sentence]
        for word, tag, tag_history in zip(words, tags, self.history_generator(tags)):
            self.c_tw[tag,word] += 1
            self.c_t[tag] += 1
            for suffix in suffixes(tag_history):
                self.c_h[suffix] += 1
                self.c_ht[suffix, tag] += 1
            self.word_lexicon[word].add(tag)
            self.tag_lexicon[tag].add(word)

    def train_lambdas(self, held_out_data):
        held_out_data = list(held_out_data)
        
        epsilon = 0.001

        # Debug output
        print("Starting Smoothing EM Algorithm", file=sys.stderr)
        print("Actual Lambdas:", self.lambdas, file=sys.stderr)
        print("Actual Cross Entropy:", self.tag_cross_entropy(held_out_data), file=sys.stderr)

        done = False
        iteration = 0
        while not done:
            iteration += 1
            print("\nStarting iteration %s" % iteration, file=sys.stderr)

            # Prepare list of lambda multiplicators
            lambdas = [0 for _ in self.lambdas]

            for sentence in held_out_data:
                tags = [tag for word,tag in sentence]
                for tag, tag_history in zip(tags, self.history_generator(tags)):
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
            print("New Cross Entropy:", self.tag_cross_entropy(held_out_data), file=sys.stderr)

        print("End of EM Smoothing Algorithm", file=sys.stderr)

    def tag_cross_entropy(self, held_out_data):
        sum = 0
        count = 0
        for sentence in held_out_data:
            tags = [tag for word,tag in sentence]
            for tag, tag_history in zip(tags, self.history_generator(tags)):
                sum += self.log_tag_probability(tag, tag_history)
                count += 1
        return -sum / count

    def decode(self, data):
        for sentence in data:
            yield self.decode_sentence(sentence)

    def decode_sentence(self, sentence):
        words = list(sentence)

        max_number_of_states_in_stage = 50
        
        print("Decoding sentence", sentence, file=sys.stderr)

        # Viterbi algorithm
        stage = { self.start_state() : ViterbiTrelisNode(log_gamma=0) }
        for word in words:
            new_stage = defaultdict(ViterbiTrelisNode)
            for previous_state, previous_node in stage.items():
                for tag in self.possible_tags(word):

                    # Computing gamma when comming from the previous node
                    log_gamma = previous_node.log_gamma \
                            + self.log_tag_probability(tag, previous_state) \
                            + self.log_word_probability(word, tag)

                    # Updating the state
                    state = previous_state[1:] + (tag,)
                    new_stage[state].update_node(log_gamma, previous_node, tag)

            # pruning
            pruned = heapq.nlargest(max_number_of_states_in_stage, new_stage.items(), key=lambda x: x[1].log_gamma)
            stage = dict(pruned)

        # Backtrace the best tag path
        node = max(stage.values(), key=lambda x: x.log_gamma)
        rev_tags = []
        while node.tag is not None:
            rev_tags.append(node.tag)
            node = node.previous_node
        return list(zip(words,reversed(rev_tags)))

    
    def possible_tags(self, word):
        tags = self.word_lexicon[word]
        return tags if tags else self.tag_lexicon.keys()
        #return self.tag_lexicon.keys()

    def vocabulary_size(self, tag=None):
        if tag is None:
            return len(self.word_lexicon)
        else:
            return len(self.tag_lexicon[tag])


    def log_tag_probability(self, tag, tag_history):
        return log2(self.tag_probability(tag, tag_history))

    def log_word_probability(self, word, tag):
        return log2(self.word_probability(word, tag))

    def start_state(self):
        return (self.n - 1) * (None,)

    def history_generator(self, iterable):
        iterator = iter(iterable)
        n_gram = deque(self.start_state())
        while True:
            new_item = next(iterator)
            yield tuple(n_gram)
            n_gram.popleft()
            n_gram.append(new_item)
    
    def tag_probability(self, tag, tag_history):
        sum = 0
        for lambda_coeff, suffix in safe_zip(self.lambdas[:-1], suffixes(tag_history)):
            sum += lambda_coeff * self.n_tag_probability(tag, suffix)
        sum += self.lambdas[-1] / self.vocabulary_size()
        return sum

    def n_tag_probability(self, tag, suffix):
        try:
            return self.c_ht[suffix, tag] / self.c_h[suffix] 
        except ZeroDivisionError:
            return 1/self.vocabulary_size()

    def word_probability(self, word, tag):
        nominator = self.c_tw[tag, word] + 1
        denominator = self.c_t[tag] + self.vocabulary_size(tag)
        return nominator / denominator

class ViterbiTrelisNode(object):
    __slots__ = ["log_gamma", "previous_node", "tag"]
    def __init__(self, log_gamma=None, previous_node=None, tag=None):
        self.log_gamma = log_gamma
        self.previous_node = previous_node
        self.tag = tag

    def update_node(self, log_gamma, previous_node, tag):
        if self.log_gamma is None or self.log_gamma < log_gamma:
            self.log_gamma = log_gamma
            self.previous_node = previous_node
            self.tag = tag
        
class SupervisedHMMTagger(HMMTagger):
    pass

class UnsupervisedHMMTagger(HMMTagger):
    def train_unlabeled(self, unlabeled_sentences):
        # Forward-Backward algorithm

        unlabeled_sentences = list(unlabeled_sentences)
        
        TrelisNode = namedtuple("TrelisNode", ["log_alpha","log_beta"])

        done = False
        while not done:
            counts3 = Counter()
            counts2 = Counter()
            counts1 = Counter()

            for sentence in unlabeled_sentences:
                # Compute forward probabilities (alphas)
                stages = [ ]


        



def safe_zip(*args):
    for n_tuple in zip_longest(*args):
        assert None not in n_tuple
        yield n_tuple

def suffixes(sequence):
    for i in range(len(sequence) + 1):
        yield sequence[i:]



