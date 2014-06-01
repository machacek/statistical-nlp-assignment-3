#!/usr/bin/env python3
from collections import namedtuple, Counter, defaultdict, deque
from math import log2
from itertools import zip_longest
from functools import reduce
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
            print("\n\nStarting iteration %s" % iteration, file=sys.stderr)

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
            for tag in self.possible_tags(word):
                for previous_state, previous_node in stage.items():

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
        if tags:
            return tags
        else:
            #print("Unknown word: %s" % word, file=sys.stderr)
            return self.tag_lexicon.keys()

    def vocabulary_size(self, tag=None):
        if tag is None:
            return len(self.word_lexicon)
        else:
            return len(self.tag_lexicon[tag])

    def log_tag_probability(self, tag, tag_history):
        try:
            return log2(self.tag_probability(tag, tag_history))
        except ValueError:
            return float('-inf')

    def log_word_probability(self, word, tag):
        try:
            return log2(self.word_probability(word, tag))
        except ValueError:
            return float('-inf')

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
            return self.history_tag_expected_counts[suffix, tag] / self.history_expected_counts[suffix]
        except AttributeError:
            try:
                return self.c_ht[suffix, tag] / self.c_h[suffix] 
            except ZeroDivisionError:
                return 1/self.vocabulary_size()

    def word_probability(self, word, tag):
        try:
            nominator = self.tag_word_expected_counts[tag,word]
            denominator = self.tag_expected_counts[tag]
            return nominator / denominator
        except AttributeError:
            nominator = self.c_tw[tag, word] + 1
            denominator = self.c_t[tag] + self.vocabulary_size()
            return nominator / denominator

    def train_unlabeled(self, unlabeled_sentences):
        unlabeled_sentences = list(unlabeled_sentences)


        # Forward-Backward algorithm
        for i in range(10):
            print("\n\nForward-Backward algorithm - iteration %s" % i, file=sys.stderr)

            tag_word_expected_counts = Counter()
            tag_expected_counts = Counter()
            history_tag_expected_counts = Counter()
            history_expected_counts = Counter()

            corpus_log_prob = 0

            for sentence in unlabeled_sentences:
                # print('.', end='', file=sys.stderr)
                sys.stderr.flush()

                # Compute forward probabilities (alphas)
                stages = [{ self.start_state() : ForwardBackwardTrelisNode(log_alpha=0) }]
                for word in sentence:
                    new_stage = defaultdict(ForwardBackwardTrelisNode)
                    
                    for tag in self.possible_tags(word):
                        for previous_state, previous_node in stages[-1].items():
                            # Computint alpha increase
                            log_alpha_inc = previous_node.log_alpha \
                                    + self.log_tag_probability(tag, previous_state) \
                                    + self.log_word_probability(word, tag)
                            
                            # Increasing the alpha
                            new_state = previous_state[1:] + (tag,)
                            new_stage[new_state].log_inc_alpha(log_alpha_inc)

                    # Adding the new stage to the list
                    stages.append(new_stage)


                # Compute backward probabilities (betas)
                for node in stages[-1].values():
                    node.log_beta=0
                for t, word in reversed(list(enumerate(sentence))):
                    current_stage = stages[t]
                    next_stage = stages[t+1]
                    for tag in self.possible_tags(word):
                        for state, node in current_stage.items():

                            # Getting next stage node for given tag, to which there is an edge from actual node
                            next_state = state[1:] + (tag,)
                            next_node = next_stage[next_state]

                            # Computing beta increase
                            log_beta_inc = next_node.log_beta \
                                    + self.log_word_probability(word, tag) \
                                    + self.log_tag_probability(tag, state)

                            node.log_inc_beta(log_beta_inc)
                
                
                # Checkpoints
                # for i, stage in enumerate(stages):
                #     print(i, reduce(log_add, (node.log_alpha + node.log_beta for node in stage.values())), file=sys.stderr)
                sentence_log_prob = reduce(log_add, (node.log_alpha for node in stages[-1].values()))
                corpus_log_prob += sentence_log_prob
                print("Sentence log probability: %s" % sentence_log_prob, file=sys.stderr)

                # Accumulate the counts
                for t, word in enumerate(sentence, 1):
                    for tag in self.possible_tags(word):
                        for previous_state, previous_node in stages[t-1].items():
                            state = previous_state[1:] + (tag,)
                            node = stages[t][state]

                            #print(previous_node.log_alpha, node.log_beta, file=sys.stderr)
                            
                            log_expected_count_inc = \
                                      previous_node.log_alpha \
                                    + self.log_tag_probability(tag, previous_state) \
                                    + self.log_word_probability(word, tag) \
                                    + node.log_beta

                            # print(log_expected_count_inc, file=sys.stderr)

                            tag_word_expected_counts[tag,word] += 2**(log_expected_count_inc / len(sentence))
                            tag_expected_counts[tag] += 2**(log_expected_count_inc / len(sentence))

                            history = previous_state
                            for suffix in suffixes(history):
                                history_tag_expected_counts[suffix, tag] += 2**(log_expected_count_inc / len(sentence))
                                history_expected_counts[suffix] += 2**(log_expected_count_inc / len(sentence))

            # Substitute current counts
            self.tag_word_expected_counts = tag_word_expected_counts
            self.tag_expected_counts = tag_expected_counts
            self.history_tag_expected_counts = history_tag_expected_counts
            self.history_expected_counts = history_expected_counts

            tag_word_sums = Counter()
            history_tag_sums = Counter()

            print("Corpus log probability: %s" % corpus_log_prob, file=sys.stderr)


                
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

class ForwardBackwardTrelisNode(object):
    __slots__ = ["log_alpha", "log_beta"]
    def __init__(self, log_alpha=float('-inf'), log_beta=float('-inf')):
        self.log_alpha = log_alpha
        self.log_beta = log_beta

    def log_inc_alpha(self, log_alpha_addition):
        self.log_alpha = log_add(self.log_alpha, log_alpha_addition)
    
    def log_inc_beta(self, log_beta_addition):
        self.log_beta = log_add(self.log_beta, log_beta_addition)

def safe_zip(*args):
    for n_tuple in zip_longest(*args):
        assert None not in n_tuple
        yield n_tuple

def suffixes(sequence):
    for i in range(len(sequence) + 1):
        yield sequence[i:]

log_big = log2(10**30)
def log_add(x,y):
    if x == float('-inf') and y == float('-inf'):
        return y
    elif y - x > log_big:
        return y
    elif x - y > log_big:
        return x
    else:
        _min = min(x,y)
        return _min + log2(2**(x - _min) + 2**(y - _min))
