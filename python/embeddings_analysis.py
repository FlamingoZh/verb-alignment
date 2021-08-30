import pickle
import numpy as np
from numpy import unique
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.cluster import MiniBatchKMeans, KMeans
from munkres import Munkres, print_matrix

from utils.utils_funcs import flatten_pkl,dimention_reduction_TSNE,gen_distinct_colors,search_index



def gen_visualizing_data(data,word_list,word_order,n_sample):

    index_list=search_index(word_list,word_order)
    visualizing_data = list()
    for i in range(n_sample):
        temp = [data[i]['z_0'][k] for k in index_list]
        visualizing_data.extend(temp)
    return np.array(visualizing_data)


def construct_list_data(data, picked_concepts, n_sample_per_concept, z='z_0',start_index=0):
    if "L_words" in data[0].keys():
        words = data[0]["L_words"]
    else:
        words = data[0]["vocab_intersect"]

    index_list = search_index(picked_concepts, words)

    constructed_data = list()
    for sample in data[start_index:start_index + n_sample_per_concept]:
        sample_list = [sample[z][k] for k in index_list]
        for j in range(len(sample_list)):
            temp = dict()
            temp["embedding"] = np.array(sample_list[j])
            temp["true_label"] = words[index_list[j]]
            temp["true_label_index"] = j
            constructed_data.append(temp)
    return constructed_data


def unsupervised_clustering(data, n_clusters, method, labeled_data=None):
    x = [element['embedding'] for element in data]

    # define and fit model
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters)
        model.fit(x)
    elif method == "minibatchkmeans":
        model = MiniBatchKMeans(n_clusters=n_clusters)
        model.fit(x)
    # elif method == "mykmeans":
    #     model = MyKMeans(n_clusters=n_clusters)
    #     model.fit(x, labeled_data)
    # elif method == "naive classfication":
    #     model = NaiveClassfication(n_clusters=n_clusters)
    #     model.fit(x, labeled_data)

    # assign a cluster to each example
    yhat = model.predict(x)

    # reconstruct data sturcture
    constructed_data = list()
    for i in range(len(x)):
        temp = dict()
        temp['embedding'] = np.array(data[i]['embedding'])
        temp['true_label'] = data[i]['true_label']
        temp["true_label_index"] = data[i]['true_label_index']
        temp['pred_label'] = yhat[i]
        constructed_data.append(temp)
    return constructed_data


# manual_pick_vg=gen_visualizing_data(vg_verb_data,manual_pick,vg_verb,n_sample)
# print(manual_pick_vg.shape)
# manual_pick_mit=gen_visualizing_data(mit_verb_data,manual_pick,mit_verb,n_sample)
# print(manual_pick_mit.shape)

def visualization_cluster(title,data, colors, label, perplexity=30, print_y=None):
    if label == "pred":
        y = [element['pred_label'] for element in data]
        clusters = unique(y)
    elif label == "true":
        y = [element['true_label'] for element in data]
        clusters = unique(y)
    else:
        print("wrong label category")
        return None

    if print_y:
        print("y:", y)

    embeds = [element['embedding'] for element in data]
    embeds_reduced = dimention_reduction_TSNE(embeds, perplexity)

    # create scatter plot for samples from each cluster

    fig, ax = plt.subplots(1, 1)
    ax.set_title(title)

    for i in range(len(clusters)):
        # get row indexes for samples with this cluster
        row_ix = list()
        for j in range(len(y)):
            if y[j] == clusters[i]:
                row_ix.append(j)

        # color control
        col_mat = list()
        for ii in range(len(row_ix)):
            col_mat.append(colors[i])
        col_mat = np.vstack(col_mat)

        ax.scatter(embeds_reduced[row_ix, 0], embeds_reduced[row_ix, 1], c=col_mat,label=str(clusters[i]))
    plt.legend(bbox_to_anchor=(1.05,0),loc=3,borderaxespad=0)
    plt.tight_layout()
    plt.show()




# vg_tsne=np.array(TSNE(n_components=2,perplexity=5).fit_transform(manual_pick_vg))
# vg_label=[val for val in range(8) for i in range(5)]
# colors=['red','blue','yellow','pink','black','grey','purple','cyan']
#
# fig = plt.figure(figsize=(8,8))
# plt.scatter(vg_tsne[:,0], vg_tsne[:,1], c=vg_label, cmap=matplotlib.colors.ListedColormap(colors))
# plt.show()



# mit_tsne=np.array(TSNE(n_components=2,perplexity=3).fit_transform(manual_pick_mit))
# mit_label=[i for val in range(n_sample) for i in range(8)]
# print(mit_label)
# colors=['red','blue','yellow','pink','black','grey','purple','cyan']
#
# fig = plt.figure(figsize=(8,8))
# plt.scatter(mit_tsne[:,0], mit_tsne[:,1], c=mit_label, cmap=matplotlib.colors.ListedColormap(colors))
# plt.show()

# def visualize_embedding(name,data,n_concept,n_sample,perplexity=30):
#     assert data.shape[0]==n_concept*n_sample
#     data_tsne = np.array(dimention_reduction_TSNE(data))
#     data_label=[i for j in range(n_sample) for i in range(n_concept)]
#     color_list = plt.cm.Set3(np.linspace(0, 1, n_concept))
#     fig, ax = plt.subplots(1, 1,figsize=(8, 8))
#     ax.set_title(name)
#     ax.scatter(data_tsne[:, 0], data_tsne[:, 1], c=data_label, cmap=matplotlib.colors.ListedColormap(color_list))
#     plt.show()
#
# visualize_embedding("MiT",manual_pick_mit,8,n_sample,5)
# visualize_embedding("VG",manual_pick_vg,8,n_sample,5)


def label_mapping(data, n_category, verbose=False):
    # clusters = unique([ele["true_label"] for ele in data])
    # true_label_to_idx = dict()
    # for i, cluster in enumerate(clusters):
    #     true_label_to_idx[cluster] = i
    # if verbose:
    #     print(true_label_to_idx)

    freq_mat = np.zeros((n_category, n_category), dtype=int)
    for item in data:
        true_label = item["true_label_index"]
        pred_label = item["pred_label"]
        freq_mat[true_label][pred_label] += 1

    cost_matrix = []
    for row in freq_mat:
        cost_row = []
        for col in row:
            cost_row += [100000000 - col]
        cost_matrix += [cost_row]

    m = Munkres()
    indexes = m.compute(cost_matrix)
    if verbose:
        print_matrix(freq_mat, msg='Highest profit through this matrix:')
        total = 0
        for row, column in indexes:
            value = freq_mat[row][column]
            total += value
            print('(%d, %d) -> %d' % (row, column, value))
        print('total profit: %d' % total)

    pred_label_map = dict()
    for e in indexes:
        pred_label_map[e[1]] = e[0]
    return pred_label_map


def compute_f1_score(data,pred_label_map,verbose=False):
    if verbose:
        print("true_label:\n",[ele["true_label"] for ele in data])
        print("pred_label:\n",[ele["pred_label"] for ele in data])

    unified_true_label=[ele["true_label_index"] for ele in data]
    unified_pred_label=[pred_label_map[ele["pred_label"]] for ele in data]

    if verbose:
        print("\naligned_true_label:\n",unified_true_label)
        print("aligned_pred_label:\n",unified_pred_label)

    print("\nf1_score(macro):",f1_score(unified_true_label, unified_pred_label, average='macro'))
    print("f1_score(micro):",f1_score(unified_true_label, unified_pred_label, average='micro'))

# bias=0
# basic=manual_pick_mit[bias]
# for val in range(bias,8*n_sample,1):
#     dist=distance.euclidean(basic,manual_pick_mit[val])
#     print(dist)
# print("-------------------")
# for val in range(bias+1,8*n_sample,8):
#     dist=distance.euclidean(basic,manual_pick_mit[val])
#     print(dist)


if __name__ == '__main__':
    vg_verb_data = flatten_pkl(pickle.load(open("../data/dumped_data/vg_verb_25samples_10blocks.pkl", "rb")))
    mit_verb_data = pickle.load(open("../data/dumped_data/MiT_swav_100.pkl", "rb"))
    vg_noun_data = flatten_pkl(pickle.load(open("../data/dumped_data/vg_noun_25samples_10blocks.pkl", "rb")))

    vg_verb = vg_verb_data[0]['vocab_intersect']
    mit_verb = mit_verb_data[0]['L_words']

    intersect_verb = set(vg_verb).intersection(set(mit_verb))
    print("intersect verb:", len(intersect_verb), intersect_verb)

    #print(vg_noun_data[0]['vocab_intersect'])

    manual_pick = ['shake', 'drive', 'chase', 'lean', 'buy', 'pick', 'bake', 'climb']
    manual_pick_noun=['airplane','animal','apple','baby','arm','bag','balcony','ball']
    n_concept = len(manual_pick)
    n_sample_per_concept = 5

    picked_data_vg = construct_list_data(vg_verb_data, manual_pick, n_sample_per_concept)
    picked_data_vg_noun = construct_list_data(vg_noun_data, manual_pick_noun, n_sample_per_concept)
    picked_data_mit = construct_list_data(mit_verb_data, manual_pick, n_sample_per_concept)

    u_picked_data_vg = unsupervised_clustering(picked_data_vg, n_concept, 'kmeans')

    visualization_cluster("VG Verb", picked_data_vg, gen_distinct_colors(n_concept), label='true', perplexity=5)
    visualization_cluster("VG Noun", picked_data_vg_noun, gen_distinct_colors(n_concept), label='true', perplexity=5)
    visualization_cluster("MiT",picked_data_mit,gen_distinct_colors(n_concept),label='true',perplexity=5)

    #visualization_cluster("VG prediction", u_picked_data_vg, gen_distinct_colors(n_concept), label='pred', perplexity=5)

    print("-----------------------")
    pred_label_map = label_mapping(u_picked_data_vg, n_concept, verbose=False)
    compute_f1_score(u_picked_data_vg, pred_label_map, verbose=False)




# vg_index=list()
# mit_index=list()
# for item in manual_pick:
#     vg_index.append(vg_verb.index(item))
#     mit_index.append(mit_verb.index(item))
#
#
# manual_pick_vg=list()
# for i in range(n_sample):
#     temp=[vg_verb_data[i]['z_0'][k] for k in vg_index]
#     manual_pick_vg.extend(temp)
# manual_pick_vg=np.array(manual_pick_vg)
# print(manual_pick_vg.shape)
#
# manual_pick_mit=list()
# for i in range(n_sample):
#     temp=[mit_verb_data[i]['z_0'][k] for k in mit_index]
#     manual_pick_mit.extend(temp)
# manual_pick_mit=np.array(manual_pick_mit)
# print(manual_pick_mit.shape)



