# A Common Framework for Quantifying the Learnability of Nouns and Verbs

This code provides implementations of some analyses in the paper [A Common Framework for Quantifying the Learnability of Nouns and Verbs](https://escholarship.org/uc/item/8dn6k82j).

## Recommended Python Environment

Python==3.8
Pytorch>=1.8
scikit-learn>=0.24
matplotlib>=3.4.2

## Getting Started

1. Clone the code from github:
```
	git clone https://github.com/FlamingoZh/verb-alignment.git
    cd verb-alignment 
```

Generate samples of visual and language embeddings for faster future computation:

	python python/gen_data.py

Data will be stored in pkl file format (you may refer to [Python pickle module](https://docs.python.org/3/library/pickle.html) for detailed guidance on usage).

Run simulations to check the alignability of language embeddings and embeddings generated from video:

	python python/analyses_video.py

Vary the number of videos Run multiple simulations



## Advanced Usage

