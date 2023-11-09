import random
import argparse
import codecs
import os
import numpy
import numpy as np


# observations
class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    START_STATE = "#"  # TODO: add dependency injection in constructor

    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        # Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}
        self.transitions = transitions
        self.emissions = emissions

    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        filenames = [basename + '.trans', basename + '.emit']
        for filename in filenames:
            with open(filename, 'r') as file:
                lines = file.read().split('\n')
                d = {}
                for line in lines:
                    kkv = line.split(' ')
                    if kkv[0] in d.keys():
                        d[kkv[0]].update({kkv[1]: float(kkv[2])})
                    else:
                        d[kkv[0]] = {kkv[1]: float(kkv[2])}
            if filename.endswith('.trans'):
                self.transitions = d
            else:
                self.emissions = d

    # you do this.
    def generate(self, n):
        """return an n-length observation by randomly sampling from this HMM."""
        hidden_states = list(self.transitions[self.START_STATE].keys())
        successors = []
        emissions = []
        for i in range(n):
            successor = random.choice(hidden_states)
            emissions_options = list(self.emissions[successor].keys())
            emission = random.choice(emissions_options)
            successors.append(successor)
            emissions.append(emission)
        o = Observation(successors, emissions)
        return o

    def load_observation(self, filename):
        with open(filename, 'r') as file:
            observations = []
            lines = file.read().split('\n\n')
            for line in lines:
                if line == '':
                    continue
                emissions = line.strip().split(' ')
                o = Observation([], emissions)
                observations.append(o)
            return observations

    def forward(self, observation):
        states = list(self.transitions[self.START_STATE].keys())
        seq = observation.outputseq
        rows = len(states) + 1
        cols = len(observation)
        M = [[0 for i in range(cols)] for j in range(rows)]

        # dp tabulation of probabilities
        for i, s in enumerate(states):
            start_prob = self.transitions[self.START_STATE][s]
            prob_given_obsrv = self.emissions[s].get(seq[0], 0.0)
            M[i][0] = start_prob * prob_given_obsrv
        for i in range(1, cols):
            for s in states:
                for k, s2 in enumerate(states):
                    M[k][i] += M[k][i-1] * self.transitions[s2][s] * self.emissions[s].get(seq[i], 0.0)

        # find and return most probable state
        last_emission = [row[-1] for row in M]
        state_prob = np.max(last_emission)
        return states[last_emission.index(state_prob)]


    # you do this: Implement the Viterbi algorithm. Given an Observation (a list of outputs or emissions)
    # determine the most likely sequence of states.
    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", type=str)
    parser.add_argument("--forward", type=str)
    parser.add_argument("--viterbi", type=str)

    args = parser.parse_args()
    hmm = HMM()
    hmm.load('partofspeech.browntags.trained')  # TODO: get flag value

    if args.forward:
        pass  # TODO
    if args.viterbi:
        pass  # TODO
    if args.generate:
        pass  # TODO


