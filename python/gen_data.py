from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import random
import pickle
import torch.optim
import torch.nn.parallel

import pprint

from utils.utils_funcs import loadGlove

from mit_video import video_models
from mit_video.video_utils import extract_frames

import utils.resnet50 as resnet_models

def lemmatize(words):
    new_words=list()
    for word in words:
        if "singing" in word:
            word='singing'
        elif "speaking" in word:
            word="speaking"
        elif "playing" in word:
            word="playing"
        new_words.append(word)
    new_words=sorted(list(set(new_words))) # remove duplicates
    L = WordNetLemmatizer()
    L_words=list()
    for word in new_words:
        L_word=L.lemmatize(word,"v")
        L_words.append(L_word)
    return new_words,L_words

def process_video(words,model_name):
    print("Loading video model "+model_name+"...")
    if model_name=='resnet3d50':
        model = video_models.load_model('resnet3d50')
    elif model_name=='swav':
        model = resnet_models.__dict__['resnet50'](
            normalize=True,
            hidden_mlp=2048,
            output_dim=128,
            nmb_prototypes=3000,
        )
        cc = torch.load('../data/swav_800ep_pretrain.pth.tar', map_location="cpu")
        model_state_dict = {}
        for key in cc:
            model_state_dict[key[7:]] = cc[key]
            # print(key[7:],key)
        model.load_state_dict(model_state_dict)
    else:
        print('unknown video model, select resnet3d50 as default.')
        model = video_models.load_model('resnet3d50')
    model.eval()
    transform = video_models.load_transform()
    print("Finish!")

    base_path = 'D:/moments_data/Moments_in_Time_Raw/training/'
    video_embeddings_list=list()
    for i,word in enumerate(words):
        path = ''.join([base_path, word, "/"])
        file_names = list()
        dirs = os.listdir(path)
        if not dirs:
            print(path, "doesn't contain videos")
        else:
            for x in dirs:
                if x.endswith('mp4'):
                    file_names.append(x)
            # if i%10==0:
            #     print(i,"/",len(words))
            flag=False
            while not flag:
                try:
                    selected_video = random.sample(file_names, k=1)
                    video_path="".join([path,selected_video[0]])
                    frames = extract_frames(video_path, 16)
                    if model_name=='resnet3d50':
                        transformed_frame = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
                        with torch.no_grad():
                            embedding = model(transformed_frame)
                    elif model_name=='swav':
                        transformed_frame = torch.stack([transform(frame) for frame in frames], 0)
                        with torch.no_grad():
                            embedding,_ = model(transformed_frame)
                        embedding=torch.mean(embedding,0).unsqueeze(0)
                    else:
                        # default resnet3d50 configurations
                        transformed_frame = torch.stack([transform(frame) for frame in frames], 1).unsqueeze(0)
                        with torch.no_grad():
                            embedding = model(transformed_frame)
                    flag=True
                except:
                    print('error:',video_path)
                    flag=False
            video_embeddings_list.append(torch.squeeze(embedding).numpy())
        # if i>5:
        #     break

    video_embeddings=np.array(video_embeddings_list)
    print("Finish processing videos.")
    return video_embeddings

def gen_embeddings(video_model_name,n_image=1,dump=False):
    words = list()
    with open("../data/category_momentsv2.txt", 'r', encoding="utf-8") as f:
        for line in f:
            word = line[:-1]
            words.append(word)
    new_words,L_words=lemmatize(words)
    # if dump:
    #     with open('../data/video_truncated_words.txt', 'w') as f:
    #         for line in new_words:
    #             f.write(line+"\n")
    #     with open('../data/video_lemmatized_words.txt', 'w') as f:
    #         for line in L_words:
    #             f.write(line + "\n")

    return_data=list()

    for i_image in range(n_image):
        print("image:",i_image+1,"/",n_image)
        # z_0
        z_0=process_video(new_words,video_model_name)
        #print(z_0.shape)

        # z_1
        print("Loading word model...")
        glove_dict = loadGlove('../data/glove.840B.300d.txt')
        print("Finish!")
        z_1_list=list()
        for i,word in enumerate(L_words):
            z_1_embed=glove_dict[word]
            z_1_list.append(z_1_embed)
            # if i>5:
            #     break
        z_1=np.array(z_1_list)
        #print(z_1.shape)
        return_data.append(dict(z_0=z_0,z_1=z_1,vocab_intersect=new_words,L_words=L_words))

    if dump:
        pickle.dump(return_data,open('../data/dumped_data/MiT_'+video_model_name+'_'+str(n_image)+'.pkl','wb'))
        print("Embeddings dumped to", 'data/dumped_data/MiT_'+video_model_name+'_'+str(n_image)+'.pkl')

    return return_data



# generate data and dump them
#gen_embeddings('resnet3d50',n_image=100,dump=True)

#gen_embeddings('resnet3d50',n_image=1,dump=False)
gen_embeddings('swav',n_image=1,dump=True)
