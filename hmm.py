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
        self.word_lexicon = defaultdict(set)
        self.tag_lexicon = defaultdict(set)
        self.transition_probs = dict()
        self.output_probs = dict()

    def train_labeled(self, sentences):
        train_data = list(self.concat_labeled_sentences(sentences))
        words = [word for word, tag in train_data]
        tags = [tag for word, tag in train_data]

        transition_probs = defaultdict(Distribution)
        output_probs = defaultdict(Distribution)

        for word, tag, state in zip(words, tags, self.history_generator(tags)):
            output_probs[tag].add_count(word, 1)
            for history in suffixes(state):
                transition_probs[history].add_count(tag, 1)
            self.word_lexicon[word].add(tag)
            self.tag_lexicon[tag].add(word)

        # Smoothing output probabilities
        # options:
        #    - not do
        #    * all possible pairs +1
        #    - train lambdas like transition smoothing
        for word in self.word_lexicon:
            for tag in self.tag_lexicon:
                output_probs[tag].add_count(word, 1)

        self.transition_probs.update(transition_probs)
        self.output_probs.update(output_probs)
    
    def train_unlabeled(self, unlabeled_sentences):
        unlabeled_data = list(self.concat_unlabeled_sentences(unlabeled_sentences))

        # Forward-Backward algorithm
        for i in range(10):
            print("\nForward-Backward algorithm - iteration %s" % i, file=sys.stderr)

            transition_probs = defaultdict(Distribution)
            output_probs = defaultdict(Distribution)

            # Compute forward probabilities (alphas)
            print("Computing forward probabilities", file=sys.stderr)
            stages = [{ self.start_state() : ForwardBackwardTrelisNode(log_alpha=0) }]
            for word in unlabeled_data:
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
            print("Computing backward probabilities", file=sys.stderr)
            for node in stages[-1].values():
                node.log_beta=0
            for t, word in reversed(list(enumerate(unlabeled_data))):
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
            
            data_log_prob = log_sum(node.log_alpha for node in stages[-1].values())
            print("Data log probability: %s" % data_log_prob, file=sys.stderr)

            # Accumulate the counts
            print("Accumulating counts", file=sys.stderr)
            for t, word in enumerate(unlabeled_data, 1):
                for tag in self.possible_tags(word):
                    for previous_state, previous_node in stages[t-1].items():
                        state = previous_state[1:] + (tag,)
                        node = stages[t][state]
                        
                        log_expected_count = \
                                  previous_node.log_alpha \
                                + self.log_tag_probability(tag, previous_state) \
                                + self.log_word_probability(word, tag) \
                                + node.log_beta

                        output_probs[tag].add_log_count(word, log_expected_count)

                        for history in suffixes(previous_state):
                            transition_probs[history].add_log_count(tag, log_expected_count)

            # Update new parameters
            self.output_probs.update(output_probs)
            self.transition_probs.update(transition_probs)

    def train_lambdas(self, sentences):
        held_out_data = list(self.concat_labeled_sentences(sentences))
        tags = [tag for word,tag in held_out_data]
        
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
            logs = [negative_infinity() for _ in self.lambdas]

            for tag, tag_history in zip(tags, self.history_generator(tags)):
                log_interpolated_prob = self.log_tag_probability(tag, tag_history)
                for i, suffix in enumerate(suffixes(tag_history)):
                    addition = self.log_n_tag_probability(tag, suffix) - log_interpolated_prob
                    logs[i] = log_add(logs[i], addition)
                addition = -log2(self.vocabulary_size) - log_interpolated_prob
                logs[-1] = log_add(logs[-1], addition)
            
            # Multiply with old lambdas and normalize
            logs = [log2(lambda_) + log for lambda_, log in zip(self.lambdas, logs)]
            log_sum_ = log_sum(logs)
            lambdas = [ 2**(log - log_sum_) for log in logs]

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

    def tag_cross_entropy(self, held_out_data):
        sum = 0
        count = 0
        tags = [tag for word,tag in held_out_data]
        for tag, tag_history in zip(tags, self.history_generator(tags)):
            sum += self.log_tag_probability(tag, tag_history)
            count += 1
        return -sum / count

    def possible_tags(self, word):
        tags = self.word_lexicon[word]
        if tags:
            return tags
        else:
            return self.tag_lexicon.keys()

    def vocabulary_size(self):
        return len(self.word_lexicon)

    def tagset_size(self):
        return len(self.tag_lexicon)

    def log_tag_probability(self, tag, tag_history):
        logs = []
        for lambda_coeff, suffix in safe_zip(self.lambdas[:-1], suffixes(tag_history)):
            logs.append(log2(lambda_coeff) + self.log_n_tag_probability(tag, suffix))
        logs.append(log2(self.lambdas[-1]) - log2(self.tagset_size()))
        return log_sum(logs)
    
    def log_n_tag_probability(self, tag, suffix):
        try:
            self.transition_probs[suffix].log_probability(tag)
        except KeyError:
            return -log2(self.tagset_size())

    def log_word_probability(self, word, tag):
        try:
            return self.output_probs[tag].log_probability(word)
        except KeyError:
            return -log2(self.vocabulary_size())
    
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
    
    def concat_labeled_sentences(self, sentences):
        first = True
        for sentence in sentences:
            if not first:
                yield from [(None, None) for _ in self.start_state()]
            first = False
            yield from sentence
    
    def concat_unlabeled_sentences(self, sentences):
        first = True
        for sentence in sentences:
            if not first:
                yield from self.start_state()
            first = False
            yield from sentence

class Distribution(object):
    __slots__ = ("log_total", "log_counts")
    def __init__(self):
        self.log_counts = defaultdict(negative_infinity)
        self.log_total = negative_infinity()

    def add_count(self, item, count):
        self.add_log_count(item, log2(count))

    def add_log_count(self, item, log_count):
        self.log_counts[item] = log_add(self.log_counts[item], log_count)
        self.log_total = log_add(self.log_total, log_count)

    def log_probability(self, item):
        if self.log_total == negative_infinity():
            # For sure
            raise KeyError("The condition was not seen")
        return self.log_counts[item] - self.log_total

    def probability(self, item):
        return 2**self.log_probability(item)
                
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

def negative_infinity():
    return float('-inf')

def safe_zip(*args):
    for n_tuple in zip_longest(*args):
        assert None not in n_tuple
        yield n_tuple

def suffixes(sequence):
    for i in range(len(sequence) + 1):
        yield sequence[i:]

log_big = 100
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

def log_sum(iterable):
    return reduce(log_add, iterable)
