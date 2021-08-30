# A Common Framework for Quantifying the Learnability of Nouns and Verbs

This code provides implementations of some analyses in the paper [A Common Framework for Quantifying the Learnability of Nouns and Verbs](https://escholarship.org/uc/item/8dn6k82j).

## Getting Started

* Clone the code from github:

    git clone https://github.com/FlamingoZh/verb-alignment.git
    cd verb-alignment 

* Generate samples of visual and language embeddings for faster future computation:

	python python/gen_data.py

Data will be in .pkl file format (you may refer to [pickle module](https://docs.python.org/3/library/pickle.html) for guidance on usage)