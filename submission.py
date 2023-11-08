from HMM import HMM


# def main():
#     # TODO: make cli callable
#     hmm = HMM()
#     hmm.load('two_english')
#     print(hmm.transitions)
#     print(hmm.emissions)


if __name__ == '__main__':
    hmm = HMM()
    hmm.load('two_english')  # 'partofspeech.browntags.trained'
    print(hmm.transitions)
    print(hmm.emissions)
