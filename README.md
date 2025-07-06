# Morphosyntactic tagger with hidden Markov models (HMM)

## Description

This project implements a morphosyntactic tagger for Spanish using a hidden Markov model (HMM). The tagger is trained on the `Corpus-tagged.txt` file, which is a collection of Spanish sentences from Wikicorpus, tagged with Part-of-Speech (POS) tags from the FreeLing library. The Viterbi algorithm is used to find the most likely sequence of tags for a given sentence.
## Technologies

* Python 3.x
* Pandas
* NumPy
* Jupyter Notebook

## Setup

1.  Clone the repository:
    ```bash
    git clone [https://github.com/hectorcaraucan/morphosyntactic-tagger-hmm.git](https://github.com/hectorcaraucan/morphosyntactic-tagger-hmm.git)
    ```
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy
    ```

## Usage

The main logic of the project is in the `morphosyntactic_tagger.ipynb` notebook. You can open it and run the cells to see the tagger in action.

The notebook is divided into the following parts:

1.  **Loading the corpus**: The `Corpus-tagged.txt` file is loaded and parsed. 
2.  **Building the HMM**:
    * Calculation of emission probabilities:  `P(word|tag)`.
    * Calculation of transition probabilities: `P(tag_i|tag_{i-1})`.
3.  **Viterbi Algorithm**: Implementation of the Viterbi algorithm to find the most likely sequence of tags for a new sentence.
4.  **Tagging a new sentence**: An example of how to use the tagger to tag a new sentence.

## Project Structure

# Morphosyntactic Tagger with Hidden Markov Models (HMM)

## Description

This project implements a morphosyntactic tagger for Spanish using a Hidden Markov Model (HMM). The tagger is trained on the `Corpus-tagged.txt` file, which is a collection of Spanish sentences from Wikicorpus, tagged with Part-of-Speech (POS) tags from the FreeLing library. The Viterbi algorithm is used to find the most likely sequence of tags for a given sentence.

## Technologies

* Python 3.x
* Pandas
* NumPy
* Jupyter Notebook

## Setup

1.  Clone the repository:
    ```bash
    git clone [https://github.com/hectorcaraucan/morphosyntactic-tagger-hmm.git](https://github.com/hectorcaraucan/morphosyntactic-tagger-hmm.git)
    ```
2.  Install the required dependencies:
    ```bash
    pip install pandas numpy
    ```

## Usage

The main logic of the project is in the `morphosyntactic_tagger.ipynb` notebook. You can open it and run the cells to see the tagger in action.

The notebook is divided into the following parts:

1.  **Loading the corpus**: The `Corpus-tagged.txt` file is loaded and parsed. 
2.  **Building the HMM**:
    * Calculation of emission probabilities:  `P(word|tag)`.
    * Calculation of transition probabilities: `P(tag_i|tag_{i-1})`.
3.  **Viterbi Algorithm**: Implementation of the Viterbi algorithm to find the most likely sequence of tags for a new sentence.
4.  **Tagging a new sentence**: An example of how to use the tagger to tag a new sentence.

## Project Structure

.
├── Corpus-tagged.txt
├── HectorCaraucanResume_2025.pdf
├── Morphosyntactic tagger with hidden Markov models (HMM).pdf
├── morphosyntactic_tagger.ipynb
├── mia07_t3_tra_resultados_emision.xlsx
├── mia07_t3_tra_resultados_trans.xlsx
└── mia07_t3_tra_resultados_viterbi.xlsx


* `Corpus-tagged.txt`: The corpus used for training the HMM. 
* `HectorCaraucanResume_2025.pdf`:  Author's resume. 
* `Morphosyntactic tagger with hidden Markov models (HMM).pdf`: Project report with theoretical background and methodology. 
* `morphosyntactic_tagger.ipynb`: Jupyter Notebook with the implementation of the HMM and the Viterbi algorithm.
* `mia07_t3_tra_resultados_emision.xlsx`: Spreadsheet with the calculated emission probabilities.
* `mia07_t3_tra_resultados_trans.xlsx`: Spreadsheet with the calculated transition probabilities.
* `mia07_t3_tra_resultados_viterbi.xlsx`: Spreadsheet with the results of the Viterbi algorithm for a test sentence.

## Methodology

### Hidden Markov Model (HMM)

A Hidden Markov Model is a statistical model that can be used to describe the generation of a sequence of observable events that depend on a sequence of unobservable (hidden) states. In this project, the observable events are the words in a sentence, and the hidden states are the morphosyntactic tags.

The HMM is defined by:
* A set of states (the POS tags).
* A set of observations (the words in the corpus).
* A matrix of transition probabilities `A`, where `A[i][j]` is the probability of transitioning from state `i` to state `j`.
* A matrix of emission probabilities `B`, where `B[j][k]` is the probability of observing `k` from state `j`.
* An initial probability distribution `π` over the states.

### Viterbi Algorithm

The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events. In the context of this project, the Viterbi algorithm is used to find the most probable sequence of POS tags for a given sentence.

## Results

The project successfully implements a morphosyntactic tagger for Spanish. The emission and transition probabilities are calculated from the provided corpus, and the Viterbi algorithm is used to tag a test sentence, showing the most likely sequence of POS tags. The results are presented in the corresponding Excel files.

## Author

* **Hector Caraucan** - [caraucan@gmail.com](mailto:caraucan@gmail.com)
