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
    hmm = HMM()
    hmm.load('partofspeech.browntags.trained')  # 'partofspeech.browntags.trained' 'two_english'

    observation = hmm.generate(20)  # n = 20, changeable
    print(observation)

    obsrv = hmm.load_observation('ambiguous_sents.obs')  # 'ambiguous_sents.obs' 'english_words.obs'
    for o in obsrv:
        print(hmm.forward(o))
    # print(hmm.forward(observation.outputseq))

    # bn_a()
    # bn_c()