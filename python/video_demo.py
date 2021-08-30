from nltk.stem import WordNetLemmatizer
import numpy as np
import pprint
import os
import random
import matplotlib.pyplot as plt
from utils.utils_funcs import loadGlove
from scipy.stats import spearmanr

import pickle

from mit_video import video_models
from mit_video.video_utils import extract_frames, load_frames, render_frames
import moviepy.editor as mpy
import torch as pt
import torch.optim
import torch.nn.parallel
from torch.nn import functional as F

from permutation import permutation
from utils.utils_funcs import plot_score_vs_accuracy_flipped
from gen_data import gen_embeddings

# print("Do permutation...")
# n_item, n_mismatch, rho_array=permutation(z_0,z_1)
# print("Draw plot.")
# draw_plot(n_item, n_mismatch, rho_array)


def draw_plot(set_size, n_mismatch_array, rho_array):
    color_ti = np.array([0, 0.4, .8])
    fontdict_tick = {'fontsize': 16}
    fontdict_label = {'fontsize': 16}
    fontdict_title = {'fontsize': 18}
    # fontdict_sub = {'fontsize': 120}
    # csfont = {'fontname':'Times New Roman'}
    csfont = {'fontname': 'Georgia'}
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Moments in Time', fontdict=fontdict_title, y=1.1, **csfont)
    ax.set_xlabel('Proportion of Mappings Permuted', fontdict=fontdict_label)
    ax.set_ylabel('Linguistic-visual Alignment', fontdict=fontdict_label)
    ax.tick_params(axis='both', labelsize=fontdict_tick['fontsize'])
    plot_score_vs_accuracy_flipped(ax, set_size, n_mismatch_array, rho_array, color_ti)
    plt.show()
    #plt.savefig("openimage.png", format='png', dpi=400, bbox_inches='tight')

# video_path="test_video.mp4"
# frames = extract_frames(video_path, 16)
# print("good")

swav_data=gen_embeddings('swav',n_image=1,dump=True)
z_0=swav_data[0]['z_0']
z_1=swav_data[0]['z_1']
set_size, n_mismatch_array, rho_array = permutation(z_0,z_1)
print("Draw plot.")
draw_plot(set_size, n_mismatch_array, rho_array)