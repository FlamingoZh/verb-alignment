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

2. Generate samples of visual and language embeddings and store them for faster future computation:
```
python python/gen_data.py
```

Data will be stored in pkl file format in `data/dumped_data/` folder (you may refer to [Python pickle module](https://docs.python.org/3/library/pickle.html) for detailed guidance on usage).

By default, the language embeddings refer to [Glove embeddings trained on Common Crawl Courpus (840B tokens)](https://nlp.stanford.edu/projects/glove/), and the visual embeddings refer to [SWaV embeddings](https://github.com/facebookresearch/swav) trained on [Moments in Time Dataset](http://moments.csail.mit.edu/).

3. Run simulations to check the alignability of language embeddings and visual embeddings:
```
python python/analyses_video.py
```

The simulation may takes several minutes to run. You would get some figure looks like below in the `figs/` folder.


4. Vary the number of visual exemplars to explore the alignment strength of the system:

```
python python/analysis_mapping-accuracy_vs_n-image.py

```

This simulation may take tens of hours to execute.

// ## Advanced Usage

// * 
