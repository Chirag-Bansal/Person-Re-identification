import os
import csv
import torch
from collections import OrderedDict

def save_network(network, epoch_label, name):
    save_filename = 'net_%s.pth'% "best"
    save_path = os.path.join('./model',name,save_filename)
    torch.save(network.cpu().state_dict(), save_path)

    if torch.cuda.is_available():
        network.cuda()

def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        label = path.split("/")[-2]
        filename = os.path.basename(path)
        camera = filename.split('_')[0]
        labels.append(int(label))
        camera_id.append(int(camera))
        # print(camera_id)
        # print(labels)
        # print(img_path)
    return camera_id, labels
