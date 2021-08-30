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
from utils.utils_funcs import average_multi

def z0_z1(data,n_sample_per_category):
    averaged_data = average_multi(random.sample(data, n_sample_per_category))
    z_0 = averaged_data['z_0']
    z_1 = averaged_data['z_1']
    return z_0,z_1


def make_plot_percent_correlation_against_n_image(datasetname,data,n_image_max=10,n_sample=5):

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
            z_0,z_1=z0_z1(data,n_image)
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
    pickle.dump(new_data,open('../data/dumped_data/percent_against_n_image_'+datasetname+'_'+str(n_image_max)+'_'+str(n_sample)+'.pkl', 'wb'))


    #ax.plot(n_image_list, percent_list, linestyle='-',label="vg_noun")
    ax.plot(n_image_list, mean_percent_list, linestyle='-')
    ax.fill_between(
        n_image_list, min_percent_list, max_percent_list,
        facecolor='green', alpha=.2, edgecolor='none'
    )

    # ax.set_yticks([0., 21., 1.])
    # ax.set_yticklabels([0.5, 1., 1.])
    #plt.legend(loc=4, ncol=1)
    plt.savefig('../figs/percent_against_n_image_'+datasetname+'_'+str(n_image_max)+'_'+str(n_sample)+'.pdf', format='pdf', dpi=400)
    plt.show()



data=pickle.load(open('../data/dumped_data/MiT_swav_100.pkl','rb'))

make_plot_percent_correlation_against_n_image('MiT_swav',data,n_image_max=25,n_sample=10)