# pos-tagging
## Part 1
Used the first 10k tagged sentences from the Brown corpus to generate the
components of a part-of-speech hidden markov model: the transition
matrix, observation matrix, and initial state distribution. Used the universal
tagset:
nltk.corpus.brown.tagged_sents(tagset=’universal’)[:10000]

## Part 2
Implemented a function viterbi() that takes arguments:
1. obs - the observations [list of ints]
2. pi - the initial state probabilities [list of floats]
3. A - the state transition probability matrix [2D numpy array]
4. B - the observation probability matrix [2D numpy array]
and returns:
1. states - the inferred state sequence [list of ints]

## Part 3
Inferred the sequence of states for sentences 10150-10152 of the Brown
corpus:
nltk.corpus.brown.tagged_sents(tagset=’universal’)[10150:10153]
