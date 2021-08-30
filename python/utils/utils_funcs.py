import os

import numpy as np
import pickle
import pathlib
from scipy.stats import spearmanr

def test():
    print("hello wolrd")

def average_multi(list_data):
    new_data=dict()
    #print(list_data[0]["z_0"].shape)
    temp1=np.zeros((list_data[0]["z_0"].shape[0],list_data[0]["z_0"].shape[1]))
    temp2=np.zeros((list_data[0]["z_1"].shape[0],list_data[0]["z_1"].shape[1]))
    for i in list_data:
        temp1+=i["z_0"]
        temp2+=i["z_1"]
    temp1=temp1/len(list_data)
    temp2=temp2/len(list_data)
    new_data["z_0"]=temp1
    new_data["z_1"]=temp2
    new_data["vocab_intersect"]=list_data[0]["vocab_intersect"]
    # If lemmatized
    if 'L_words' in list_data[0].keys():
        new_data["L_words"] = list_data[0]["L_words"]
    #print(new_data)
    return new_data

def flatten_pkl(data):
    from itertools import chain
    return list(chain.from_iterable(data))

def dimention_reduction_TSNE(vectors,perplexity=30):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0,perplexity=perplexity)
    Y = tsne.fit_transform(vectors)
    return Y

# generate n distinct colors
def gen_distinct_colors(num_colors):
    import colorsys
    colors=[]
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i/360.
        lightness = (50 + np.random.rand() * 10)/100.
        saturation = (90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return np.array(colors)

def search_index(word_list,word_order):
    index_list=list()
    for item in word_list:
        index_list.append(word_order.index(item))
    return index_list

def loadGlove(filename,dumpname='../data/embeddings_dict_840B.pkl'):
    if not pathlib.Path(dumpname).is_file():
        embeddings_dict = {}
        with open(filename, 'r', encoding="utf-8") as f:
            ii=0
            for line in f:
                ii+=1
                values = line.split()
                #print(values)
                word = values[0]
                try:
                    vector = np.asarray(values[1:], "float32")
                    embeddings_dict[word] = vector
                except:
                    print(values[0:5],ii)
        pickle.dump(embeddings_dict, open(dumpname, 'wb'))
    else:
        embeddings_dict=pickle.load(open(dumpname,'rb'))

    return embeddings_dict


def linear_interpolation(y, factor=10):
    """Interpolate additional points for piece-wise linear function."""
    n_point = len(y)
    y_interp = np.array([])
    for idx in range(1, n_point):
        # start_x = idx - 1
        # end_x = idx
        start_y = y[idx - 1]
        end_y = y[idx]
        y_interp = np.hstack((
            y_interp,
            np.linspace(start_y, end_y, factor, endpoint=False)
        ))
    y_interp = np.asarray(y_interp)
    return y_interp

def plot_score_vs_accuracy_flipped(ax, set_size, n_mismatch_array, rho_array, c):
    """"""
    accuracy_array = (set_size - n_mismatch_array) / set_size

    mismatch_list = np.unique(n_mismatch_array)
    mismatch_list = mismatch_list[1:]
    n_val = len(mismatch_list)

    loc = np.equal(n_mismatch_array, 0)
    rho_correct = rho_array[loc]
    rho_correct = rho_correct[0]

    score_mean = np.zeros(n_val)
    score_std = np.zeros(n_val)
    score_min = np.zeros(n_val)
    score_max = np.zeros(n_val)
    for idx_mismatch, val_mismatch in enumerate(mismatch_list):
        loc = np.equal(n_mismatch_array, val_mismatch)
        score_mean[idx_mismatch] = np.mean(rho_array[loc])
        score_std[idx_mismatch] = np.std(rho_array[loc])
        score_min[idx_mismatch] = np.min(rho_array[loc])
        score_max[idx_mismatch] = np.max(rho_array[loc])

    accuracy = (set_size - mismatch_list) / set_size

    rho, p_val = spearmanr(accuracy_array, rho_array)
    print('rho: {0:.2f} (p={1:.4f})'.format(rho, p_val))

    ax.plot(
        1-accuracy, score_mean, color=c,
        # marker='o', markersize=.5,
        linestyle='-', linewidth=1,
    )
    ax.fill_between(
        1-accuracy, score_mean - score_std, score_mean + score_std,
        facecolor=c, alpha=.3, edgecolor='none'
    )
    ax.fill_between(
        1-accuracy, score_min, score_max,
        facecolor=c, alpha=.25, edgecolor='none'
    )

    factor = 20
    score_beat_correct = linear_interpolation(score_max, factor=factor)
    accuracy_interp = linear_interpolation(accuracy, factor=factor)
    score_correct = rho_correct * np.ones(len(score_beat_correct))
    locs = np.less(score_beat_correct, score_correct)
    score_beat_correct[locs] = rho_correct
    ax.fill_between(
        1-accuracy_interp, score_correct, score_beat_correct,
        facecolor='r', alpha=.75, edgecolor='none'
    )
    print("rho_correct:",rho_correct)
    ax.scatter(
        0, rho_correct,
        s=100, marker='x',
        color=c
    )

    ax.set_xticks([0., .5, 1.])
    ax.set_xticklabels([0., .5, 1.])