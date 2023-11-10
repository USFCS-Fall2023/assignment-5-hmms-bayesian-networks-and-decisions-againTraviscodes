import random

from HMM import HMM
from alarm import belief_networks_alarm as bn_a
from carnet import belief_networks_carnet as bn_c


if __name__ == '__main__':
    # Part 2: Hidden Markov Models
    # Section 1: Load
    print("\nLoad")
    hmm = HMM()
    hmm.load('partofspeech.browntags.trained')
    # hmm.load('two_english')

    # Section 2: Generate
    print("\nGenerate")
    observation = hmm.generate(20)
    print(observation)

    # Section 3: Forward
    print("\nForward")
    o = hmm.load_observation('ambiguous_sents.obs')  # TODO make entire file = 1 observation
    # o = hmm.load_observation('english_words.obs')
    print(hmm.forward(o))

    # Section 4: Viterbi
    print("\nViterbi")
    print(hmm.viterbi(o))

    # Part 3: Belief Networks
    # Sections 1 & 2:
    print("\nBelief Networks: Alarm")
    bn_a()
    # Section 3:
    print("\nBelief Networks: Carnet")
    bn_c()
