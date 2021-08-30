import pickle
import numpy as np
import pprint
import random

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
from matplotlib.gridspec import GridSpec

from permutation import permutation
from utils.utils_funcs import average_multi, flatten_pkl
from embeddings_analysis import construct_list_data,unsupervised_clustering,label_mapping,compute_f1_score

def get_z0_z1_spvs_z0_and_z1(data,n_sample_per_category):
    averaged_data = average_multi(random.sample(data, n_sample_per_category))
    z_0 = averaged_data['z_0']
    z_1 = averaged_data['z_1']
    return z_0,z_1

def get_z0_z1_unspvs_z0_and_z1(data,n_sample_per_category):
    if 'L_words' in data[0].keys():
        n_concept = len(data[0]['L_words'])
    else:
        n_concept = len(data[0]['vocab_intersect'])

    sampled_data = random.sample(data, n_sample_per_category)

    if 'L_words' in data[0].keys():
        z0_data = construct_list_data(sampled_data, sampled_data[0]['L_words'], n_sample_per_category, z='z_0')
        #z1_data = construct_list_data(sampled_data, sampled_data[0]['L_words'], n_sample_per_category, z='z_1')
    else:
        z0_data = construct_list_data(sampled_data, sampled_data[0]['vocab_intersect'], n_sample_per_category, z='z_0')
        #z1_data = construct_list_data(sampled_data, sampled_data[0]['vocab_intersect'], n_sample_per_category, z='z_1')

    u_z0_data = unsupervised_clustering(z0_data, n_concept, 'kmeans')
    pred_label_map = label_mapping(u_z0_data, n_concept, verbose=False)

    unified_true_label = [ele["true_label_index"] for ele in u_z0_data]
    unified_pred_label = [pred_label_map[ele["pred_label"]] for ele in u_z0_data]
    #print(unified_true_label)
    #print(unified_pred_label)
    #compute_f1_score(u_new_data,pred_label_map)

    # print("unified_pred_label:",max(unified_pred_label),min(unified_pred_label),unified_pred_label)

    z_0 = list()
    for concept in range(n_concept):
        embed_avg_list = list()
        for index in range(len(u_z0_data)):
            if unified_pred_label[index] == concept:
                embed_avg_list.append(u_z0_data[index]['embedding'])
        embed_avg_list = np.array(embed_avg_list)
        z_0.append(np.mean(embed_avg_list, axis=0))
    z_0 = np.array(z_0)

    # z_1 = list()
    # for concept in range(n_concept):
    #     embed_avg_list = list()
    #     for index in range(len(z1_data)):
    #         if unified_true_label[index] == concept:
    #             embed_avg_list.append(z1_data[index]['embedding'])
    #     embed_avg_list = np.array(embed_avg_list)
    #     z_1.append(np.mean(embed_avg_list, axis=0))
    # z_1 = np.array(z_1)

    z_1=data[0]['z_1']

    return z_0,z_1

    # print(z_0.shape,z_1.shape)

def get_z0_z1_unspvs_z0_and_spvs_z0(data,n_sample_per_category):
    if 'L_words' in data[0].keys():
        n_concept = len(data[0]['L_words'])
    else:
        n_concept = len(data[0]['vocab_intersect'])

    sampled_data = random.sample(data, n_sample_per_category)

    if 'L_words' in data[0].keys():
        z0_data = construct_list_data(sampled_data, sampled_data[0]['L_words'], n_sample_per_category, z='z_0')
        # z1_data = construct_list_data(sampled_data, sampled_data[0]['L_words'], n_sample_per_category, z='z_1')
    else:
        z0_data = construct_list_data(sampled_data, sampled_data[0]['vocab_intersect'], n_sample_per_category, z='z_0')
        # z1_data = construct_list_data(sampled_data, sampled_data[0]['vocab_intersect'], n_sample_per_category, z='z_1')

    u_z0_data = unsupervised_clustering(z0_data, n_concept, 'kmeans')
    pred_label_map = label_mapping(u_z0_data, n_concept, verbose=False)

    unified_true_label = [ele["true_label_index"] for ele in u_z0_data]
    unified_pred_label = [pred_label_map[ele["pred_label"]] for ele in u_z0_data]
    # print(unified_true_label)
    # print(unified_pred_label)
    # compute_f1_score(u_new_data,pred_label_map)

    # print("unified_pred_label:",max(unified_pred_label),min(unified_pred_label),unified_pred_label)

    z_0 = list()
    for concept in range(n_concept):
        embed_avg_list = list()
        for index in range(len(u_z0_data)):
            if unified_pred_label[index] == concept:
                embed_avg_list.append(u_z0_data[index]['embedding'])
        embed_avg_list = np.array(embed_avg_list)
        z_0.append(np.mean(embed_avg_list, axis=0))
    z_0 = np.array(z_0)

    z_1 = average_multi(sampled_data)['z_0']

    return z_0, z_1

def draw_mapping_accuracy_against_n_image(datasetname,data,n_image_max=10,n_sample=5,z0z1_tag="language embedding and supervised visual embedding"):

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.set_xlim(0,n_image_max)
    ax.set_ylim(0,1)

    mean_percent_list=list()
    max_percent_list=list()
    min_percent_list=list()
    n_image_list=list()
    mean_percent_list.append(0)
    max_percent_list.append(0)
    min_percent_list.append(0)
    n_image_list.append(0)

    print("Start sampling...")

    for n_image in range(1,n_image_max+1):
        temp=list()
        for sample in range(1,n_sample+1):
            print("n_image, sample:",n_image,sample)

            if z0z1_tag=="language embedding and supervised visual embedding":
                z_0, z_1 =get_z0_z1_spvs_z0_and_z1(data,n_image)
            elif z0z1_tag=="language embedding and unsupervised visual embedding":
                if n_image == 1:
                    sampled_data = random.sample(data, 1)
                    z_0 = np.random.permutation(sampled_data[0]['z_0'])
                    z_1 = sampled_data[0]['z_1']
                else:
                    z_0, z_1 = get_z0_z1_unspvs_z0_and_z1(data, n_image)
            elif z0z1_tag=="supervised visual embedding and unsupervised visual embedding":
                if n_image == 1:
                    sampled_data = random.sample(data, 1)
                    z_0 = np.random.permutation(sampled_data[0]['z_0'])
                    z_1 = sampled_data[0]['z_0']
                else:
                    z_0, z_1 = get_z0_z1_unspvs_z0_and_spvs_z0(data, n_image)
            else:
                print("Fatal Error! z0z1 tag is wrong.")
                return 0

            set_size, n_mismatch_array, rho_array = permutation(z_0,z_1)
            #accuracy_array = (set_size - n_mismatch_array) / set_size

            loc = np.equal(n_mismatch_array, 0)
            # rho_correct: the alignment correlation of the true mapping
            rho_correct = rho_array[loc][0]

            # compute how many permuted systems are less alignable than the true mapping
            count=0
            for i in rho_array:
                if i<rho_correct:
                    count+=1
            percent=count/len(rho_array)
            temp.append(percent)

        print("temp:",np.max(temp),np.mean(temp),np.min(temp),temp)

        n_image_list.append(n_image)
        mean_percent_list.append(np.mean(temp))
        max_percent_list.append(np.max(temp))
        min_percent_list.append(np.min(temp))

    new_data=dict(
        n_image_list=n_image_list,
        min_percent_list=min_percent_list,
        mean_percent_list=mean_percent_list,
        max_percent_list=max_percent_list
    )
    if z0z1_tag=="language embedding and supervised visual embedding":
        file_tag='accuracy-against-n-image_'+datasetname+'_'+str(n_image_max)+'_'+str(n_sample)+'_spvs-z0-and-z1'
    elif z0z1_tag == "language embedding and unsupervised visual embedding":
        file_tag = 'accuracy-against-n-image_' + datasetname + '_' + str(n_image_max) + '_' + str(
            n_sample) + '_unspvs-z0-and-z1'
    elif z0z1_tag == "supervised visual embedding and unsupervised visual embedding":
        file_tag = 'accuracy-against-n-image_' + datasetname + '_' + str(n_image_max) + '_' + str(
            n_sample) + '_spvs-z0-and-unspvs-z0'

    pickle.dump(new_data,open('../data/dumped_data/'+file_tag+'.pkl', 'wb'))


    #ax.plot(n_image_list, percent_list, linestyle='-',label="vg_noun")
    ax.plot(n_image_list, mean_percent_list, linestyle='-')
    ax.fill_between(
        n_image_list, min_percent_list, max_percent_list,
        facecolor='green', alpha=.2, edgecolor='none'
    )

    ax.set_title(z0z1_tag, fontsize=18)

    ax.set_xlabel("Exemplars per Category",fontsize=14)
    ax.set_ylabel("Mapping Accuracy",fontsize=14)

    # ax.set_yticks([0., 21., 1.])
    # ax.set_yticklabels([0.5, 1., 1.])
    #plt.legend(loc=4, ncol=1)
    plt.tight_layout()
    plt.savefig('../figs/'+file_tag+'.png')
    plt.show()

if __name__ == '__main__':
    # data=flatten_pkl(pickle.load(open('../data/dumped_data/vg_noun_25samples_10blocks.pkl','rb')))
    # draw_alignment_strength_against_n_image('VG_Noun',data,n_image_max=15,n_sample=10)

    #data=flatten_pkl(pickle.load(open('../data/dumped_data/vg_verb_25samples_10blocks.pkl','rb')))
    #draw_alignment_strength_against_n_image('VG_Verb_swav',data,n_image_max=10,n_sample=10)

    data=pickle.load(open('../data/dumped_data/MiT_swav_100.pkl','rb'))
    draw_mapping_accuracy_against_n_image('MiT-swav',data,n_image_max=50,n_sample=10,z0z1_tag="language embedding and supervised visual embedding")

    #draw_mapping_accuracy_against_n_image('MiT-swav',data,n_image_max=10,n_sample=1, z0z1_tag="language embedding and unsupervised visual embedding")

    #draw_mapping_accuracy_against_n_image('MiT-swav', data, n_image_max=10, n_sample=1, z0z1_tag="supervised visual embedding and unsupervised visual embedding")

