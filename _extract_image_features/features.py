import json
import numpy as np
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import re
from torchvision.models import resnet152
from tqdm import tqdm


with torch.cuda.device(1):

    VQA_FOLDER = '/mnt/data/users/apodtikhov/vqa/'

    TRAIN = {'Images' : VQA_FOLDER + 'train2014/',
            'Masks' : VQA_FOLDER + 'train2014_mask/',
            'Questions' : VQA_FOLDER + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'Answers' : VQA_FOLDER + 'v2_mscoco_train2014_annotations.json',
    }

    VAL = {'Images' : VQA_FOLDER + 'val2014/',
        'Masks' : VQA_FOLDER + 'val2014_mask/',
        'Questions' : VQA_FOLDER + 'v2_OpenEnded_mscoco_val2014_questions.json',
        'Answers' : VQA_FOLDER + 'v2_mscoco_val2014_annotations.json',
    }
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_model = torch.nn.Sequential(*(list(resnet152(pretrained=True).children())[:-1]))
    image_model.eval()
    image_model.cuda()
    img_path_list = []
    for entry in os.listdir(TRAIN['Images']):
        if os.path.isfile(os.path.join(TRAIN['Images'], entry)):
            img_path_list.append(os.path.join(TRAIN['Images'], entry))

    mask_path_list = []
    for entry in os.listdir(TRAIN['Masks']):
        if os.path.isfile(os.path.join(TRAIN['Masks'], entry)):
            mask_path_list.append(os.path.join(TRAIN['Masks'], entry))

    id_to_img_path = {}
    for im in img_path_list:
        id_to_img_path[int(im.split('/')[-1].split('_')[-1].split('.')[0])] = im

    for i in tqdm(range(len(mask_path_list))):
        mask_path = mask_path_list[i]
        id = int(mask_path.split('/')[-1].split('_')[-1].split('.')[0])
        img_path = id_to_img_path[id]
        seg_num, mask = np.load(mask_path, allow_pickle=True)[()]
        img = Image.open(img_path).convert(mode='RGB')
        img = transforms.ToTensor()(img)
        mask = torch.from_numpy(mask)
        masks = []
        if img.shape[0] == 1:
            print('Hm', img_path)
        for i in range(0, seg_num + 1):
            img_seg = torch.clone(img)
            to_zero = mask != i
            img_seg[:, to_zero] = 0
            seg1 = transforms.ToPILImage()(img_seg)
            input_tensor = preprocess(seg1)
            input_batch = input_tensor.unsqueeze(0).to('cuda')
            with torch.no_grad():
                output = image_model(input_batch)
            output = output.view(1, 2048).cpu()
            masks.append(output)
        img_feat_x = torch.cat(masks, 0).numpy()
        np.save('/mnt/data/users/apodtikhov/vqa/train2014_feat/' + img_path.split('/')[-1] + '.npy', img_feat_x)

