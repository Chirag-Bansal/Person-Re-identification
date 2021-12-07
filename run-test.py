# Evaluation
# Acknowledgement: the code is based on Siddhant Kapil's repo on LA-Transformer
from __future__ import print_function

import shutup; shutup.please()

import matplotlib.pyplot as plt


import os
import faiss
import numpy as np

from PIL import Image
from tqdm.notebook import tqdm

import cv2
import timm
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from model import ClassBlock, LATransformer, LATransformerTest

from utils import get_id
from metrics import rank1, rank2, rank3, rank4, rank5, calc_ap

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ### Set feature volume sizes (height, width, depth)
# TODO: update with your model's feature length

batch_size = 1
H, W, D = 7, 7, 2048 # for dummymodel we have feature volume 7x7x2048

# ### Load Model

# TODO: Uncomment the following lines to load the Implemented and trained Model

#save_path = "<model weight path>"
#model = ReidModel(num_classes=C)
#model.load_state_dict(torch.load(save_path), strict=False)
#model.eval()

# Load ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base= vit_base.to(device)

# Create La-Transformer
model = LATransformerTest(vit_base, lmbd=8).to(device)

# Load LA-Transformer
save_path = os.path.join('/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Project/LA-Transformer/Weights/checkpoint_five.pth')
model.load_state_dict(torch.load(save_path,map_location='cpu'), strict=False)
model.eval()


# You are free to use augmentations of your own choice
transform_query_list = [
        transforms.Resize((224,224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_gallery_list = [
        transforms.Resize(size=(224,224), interpolation=3), #Image.BICUBIC
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]

data_transforms = {
        'query': transforms.Compose( transform_query_list ),
        'gallery': transforms.Compose(transform_gallery_list),
    }


image_datasets = {}
data_dir = "/home/chirag/Desktop/2021-22 Sem1/COL780: Computer Vision/Project/LA-Transformer/test-reid"

image_datasets['query'] = datasets.ImageFolder(os.path.join(data_dir, 'query'),
                                          data_transforms['query'])
image_datasets['gallery'] = datasets.ImageFolder(os.path.join(data_dir, 'gallery'),
                                          data_transforms['gallery'])
query_loader = DataLoader(dataset = image_datasets['query'], batch_size=batch_size, shuffle=False )
gallery_loader = DataLoader(dataset = image_datasets['gallery'], batch_size=batch_size, shuffle=False)

class_names = image_datasets['query'].classes


# ###  Extract Features

def extract_feature(dataloaders):

    features =  torch.FloatTensor()
    count = 0
    idx = 0
    for data in tqdm(dataloaders,total=len(dataloaders)):
        img, label = data
        # Uncomment if using GPU for inference
        #img, label = img.cuda(), label.cuda()

        output = model(img) # (B, D, H, W) --> B: batch size, HxWxD: feature volume size

        n, c, h, w = img.size()

        count += n
        features = torch.cat((features, output.detach().cpu()), 0)
        idx += 1
    return features

# Extract Query Features

query_feature= extract_feature(query_loader)

# Extract Gallery Features

gallery_feature = extract_feature(gallery_loader)

# Retrieve labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam = []
gallery_label = []
gal_id = 0
for path, v in gallery_path:
    # print(path)
    label = path.split("/")[-2]
    # print(path)
    filename = path.split("/")[-1]
    camera = filename.split('_')[0]
    gallery_label.append(int(str(gal_id)+label))
    gallery_cam.append(int(camera))
    gal_id +=1

# print(gallery_label)

query_cam = []
query_label = []
for path, v in query_path:
    # print(path)
    label = path.split("/")[-2]
    # print(path)
    filename = path.split("/")[-1]
    camera = filename.split('_')[0]
    query_label.append(int(label))
    query_cam.append(int(camera))

# gallery_cam,gallery_label = get_id(gallery_path)

# print(gallery_label)

# query_cam,query_label = get_id(query_path)

#
# # ## Concat Averaged GELTs
#
concatenated_query_vectors = []
for query in tqdm(query_feature,total=len(query_feature)):
    fnorm = torch.norm(query, p=2, dim=1, keepdim=True)*np.sqrt(14)
    query_norm = query.div(fnorm.expand_as(query))
    concatenated_query_vectors.append(query_norm.view((-1)))
#
concatenated_gallery_vectors = []
for gallery in tqdm(gallery_feature,total=len(gallery_feature)):
    fnorm = torch.norm(gallery, p=2, dim=1, keepdim=True)*np.sqrt(14)
    gallery_norm = gallery.div(fnorm.expand_as(gallery))
    concatenated_gallery_vectors.append(gallery_norm.view((-1)))
#
#
# # ## Calculate Similarity using FAISS
#
index = faiss.IndexIDMap(faiss.IndexFlatIP(10752))
#
index.add_with_ids(np.array([t.numpy() for t in concatenated_gallery_vectors]),np.array(gallery_label))
#
def search(query: str, k=1):
    encoded_query = query.unsqueeze(dim=0).numpy()
    top_k = index.search(encoded_query, k)
    return top_k
#
#
# # ### Evaluate
#
rank_vs_k = np.zeros(5)
#
rank1_score = 0
rank2_score = 0
rank3_score = 0
rank4_score = 0
rank5_score = 0
ap = 0
count = 0

query_id = 0
res_paths = []

for query, label in zip(concatenated_query_vectors, query_label):
    count += 1
    label = label
    # print(query_id, label, query_path[query_id][0])
    query_id+=1
    output_full = search(query, k=10)
    # print(output_full)
    # output2 = search2(query, k=2)
    # print(query)
    # print(output2)
    output = (output_full[0], [[int(str(i)[-3:]) for i in output_full[1][0]]])
    res_paths.append([int(str(i)[:-3]) if len(str(i)) > 3 else 0 for i in output_full[1][0]])
    # print(output)
    # output = search(query, k=10)
    rank1_score += rank1(label, output)
    rank_vs_k[0] = rank1_score
    rank2_score += rank2(label, output)
    rank_vs_k[1] = rank2_score
    rank3_score += rank3(label, output)
    rank_vs_k[2] = rank3_score
    rank4_score += rank4(label, output)
    rank_vs_k[3] = rank4_score
    rank5_score += rank5(label, output)
    rank_vs_k[4] = rank5_score
    print("Correct: {}, Total: {}, Incorrect: {}".format(rank1_score, count, count-rank1_score), end="\r")
    ap += calc_ap(label, output)

print("Rank1: %.3f, Rank5: %.3f, mAP: %.3f"%(rank1_score/len(query_feature),rank5_score/len(query_feature),
                                              ap/len(query_feature)))

print("Enter a query number")
queryid = int(input())
img = cv2.imread(query_path[queryid][0], cv2.IMREAD_UNCHANGED)
cv2.imshow("Query",img)
cv2.waitKey(0)

for i in range(5):
    img = cv2.imread(gallery_path[res_paths[queryid][i]][0], cv2.IMREAD_UNCHANGED)
    cv2.imshow("pred"+str(i+1),img)
    cv2.waitKey(0)

x = np.array([1,2,3,4,5])
rank_vs_k = np.divide(rank_vs_k,len(query_feature))

plt.title("CMC @ K")
plt.xlabel("Probability")
plt.ylabel("K")
plt.plot(x, rank_vs_k, color ="red")
plt.savefig('rank@k.png')
plt.show()
