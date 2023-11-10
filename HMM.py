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
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq)

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


# hmm model
class HMM:
    START_STATE = "#"

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
        try:
            with open(filename, 'r') as file:
                return Observation([], file.read().strip().split())
        except FileNotFoundError | FileExistsError:
            print(f"{filename} wasn't found. Please enter a different filename")

    def forward(self, observation):
        states = list(self.transitions[self.START_STATE].keys())
        seq = observation.outputseq
        rows = len(states) + 1
        cols = len(observation)
        M = [[0.0 for i in range(cols)] for j in range(rows)]

        # dp tabulation of probabilities
        for i, s in enumerate(states):
            start_prob = self.transitions[self.START_STATE][s]
            prob_given_observation = self.emissions[s].get(seq[0], 0.0)
            M[i][0] = start_prob * prob_given_observation
        for i in range(1, cols):
            for j, s in enumerate(states):
                sum = 0
                for k, s2 in enumerate(states):
                    sum += M[k][i-1] * self.transitions[s2][s] * self.emissions[s].get(seq[i], 0.0)
                M[j][i] = sum

        # find and return most probable state
        last_emission = [row[-1] for row in M]
        state_prob_idx = np.argmax(last_emission)
        return states[state_prob_idx]

    def viterbi(self, observation):
        states = list(self.transitions[self.START_STATE].keys())
        seq = observation.outputseq
        rows = len(states) + 1
        cols = len(observation)
        V = [[0.0 for i in range(cols)] for j in range(rows)]

        # dp tabulation of probabilities
        for i, s in enumerate(states):
            start_prob = self.transitions[self.START_STATE][s]
            prob_given_observation = self.emissions[s].get(seq[0], 0.0)
            V[i][0] = start_prob * prob_given_observation
        for i in range(1, cols):
            for j, s in enumerate(states):
                probabilities = [V[k][i - 1] + self.transitions[s2][s] + self.emissions[s].get(seq[i], 0.0) for k, s2 in enumerate(states)]
                V[j][i] = max(probabilities)

        # backtrace finding most probable states
        state_seq = []
        for i in range(cols, 0, -1):
            state_idx = np.argmax([row[i - 1] for row in V])
            state_seq.insert(0, states[state_idx])
        return state_seq


if __name__ == "__main__":
    '''
        To run via the command line for forward() and viterbi():
            python HMM.py <filename> --<method> <test_data_filename>
            
            e.g.: python HMM.py partofspeech.browntags.trained --forward ambiguous_sents.obs
            
        For generate():
            python HMM.py <filename> --generate <number>
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("--generate", type=int)
    parser.add_argument("--forward", type=str)
    parser.add_argument("--viterbi", type=str)

    args = parser.parse_args()
    hmm = HMM()

    if args.filename:
        hmm.load(args.filename)
    if args.forward:
        o = hmm.load_observation(args.forward)
        print(f"The most likely final state is {hmm.forward(o)}")
    if args.viterbi:
        o = hmm.load_observation(args.viterbi)
        print(' '.join(hmm.viterbi(o)))
    if args.generate:
        print(hmm.generate(args.generate))
