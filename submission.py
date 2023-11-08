import random

from HMM import HMM
from alarm import belief_networks_alarm as bn_a
from carnet import belief_networks_carnet as bn_c


# def main():
#     # TODO: make cli callable
#     hmm = HMM()
#     hmm.load('two_english')
#     print(hmm.transitions)
#     print(hmm.emissions)


if __name__ == '__main__':
    # hmm = HMM()
    # hmm.load('partofspeech.browntags.trained')  # 'partofspeech.browntags.trained' 'two_english'
    #
    # successors, emission = hmm.generate(20)  # n = 20, changeable
    # print(str(emission))

    bn_a()
    bn_c()