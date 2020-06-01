import config as my_cfg
import json
import numpy as np
import os
from PIL import Image
import preproccessing as pre
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import re
from Levenshtein import distance
import fasttext

class VQAv2(Dataset):
    def __init__(self):
        # self.fasttext_model = fasttext.load_model("wiki.en.bin")
        self.img_path_list = []
        for entry in os.listdir(my_cfg.VAL['Images']):
            if os.path.isfile(os.path.join(my_cfg.VAL['Images'], entry)):
                self.img_path_list.append(os.path.join(my_cfg.VAL['Images'], entry))
        for entry in os.listdir(my_cfg.TRAIN['Images']):
            if os.path.isfile(os.path.join(my_cfg.TRAIN['Images'], entry)):
                self.img_path_list.append(os.path.join(my_cfg.TRAIN['Images'], entry))
        self.qns_path_list = []
        for entry in os.listdir(my_cfg.VAL['Questions_feat']):
            if os.path.isfile(os.path.join(my_cfg.VAL['Questions_feat'], entry)):
                self.qns_path_list.append(os.path.join(my_cfg.VAL['Questions_feat'], entry))
        for entry in os.listdir(my_cfg.TRAIN['Questions_feat']):
            if os.path.isfile(os.path.join(my_cfg.TRAIN['Questions_feat'], entry)):
                self.qns_path_list.append(os.path.join(my_cfg.TRAIN['Questions_feat'], entry))
        
        with open(my_cfg.VAL['Questions']) as json_file:
            self.qns_list = json.load(json_file)['questions']
        with open(my_cfg.TRAIN['Questions']) as json_file:
            self.qns_list += json.load(json_file)['questions']

        with open(my_cfg.VAL['Answers']) as json_file:
            self.ans_list = json.load(json_file)['annotations'] # +=
        with open(my_cfg.TRAIN['Answers']) as json_file:
            self.ans_list += json.load(json_file)['annotations'] # +=

        self.id_to_ques = {}
        for qn in self.qns_list:
            self.id_to_ques[int(qn['question_id'])] = qn

        self.id_to_ques_path = {}
        for qn in self.qns_path_list:
            self.id_to_ques_path[int(qn.split('/')[-1].split('.')[0])] = qn

        self.id_to_img_path = {}
        for im in self.img_path_list:
            self.id_to_img_path[int(im.split('/')[-1].split('_')[-1].split('.')[0])] = im

        self.ans_to_ix, self.ix_to_ans = json.load(open(my_cfg.answer_dict, 'r'))

        with open('indexes.json') as f:
            self.types_dict = json.load(f)
        self.ans_size = len(self.ans_to_ix)
        self.data_size = len(self.ans_list)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)
        
        ans = self.ans_list[idx]
        qid = int(ans['question_id'])
        ques_path = self.id_to_ques_path[qid]
        ques_ix = np.load(ques_path)

        if ques_ix.shape[0] > my_cfg.max_token:
            sep = ques_ix[-1]
            ques_ix = ques_ix[:my_cfg.max_token]
            ques_ix[-1] = sep

        ques_ix = np.pad(
            ques_ix,
            ((0, my_cfg.max_token - ques_ix.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        id = int(ans['image_id'])
        img_path = self.id_to_img_path[id]
        if 'val' in img_path:
            texts_path = my_cfg.VAL['Texts'] + img_path.split('/')[-1].split('.')[0] + '.jpg'
            feats_path = my_cfg.VAL['Texts'][:-1] + '_1/' + img_path.split('/')[-1].split('.')[0] + '.jpg'
        else:
            texts_path = my_cfg.TRAIN['Texts'] + img_path.split('/')[-1].split('.')[0] + '.jpg'
            feats_path = my_cfg.TRAIN['Texts'][:-1] + '_1/' + img_path.split('/')[-1].split('.')[0] + '.jpg'
        texts = list(np.load(texts_path + '.npy'))
        text_feats = np.load(feats_path + '.npy')

        img_feat = np.load(img_path)

        img_feat_x = img_feat['arr_0']

        if img_feat_x.shape[0] > my_cfg.img_feat_pad_size:
            img_feat_x = img_feat_x[:my_cfg.img_feat_pad_size]

        if len(texts) > 14:
            texts = texts[:14]

        img_feat_x = np.pad(
            img_feat_x,
            ((0, my_cfg.img_feat_pad_size - img_feat_x.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )
        # Process answer
        ans_score = np.zeros(self.ans_to_ix.__len__() + 14, np.float32)
        ans_prob_dict = {}

        for ans_ in ans['answers']:
            ans_proc = pre.prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1
        
        while len(texts) < 14:
            texts.append('')
        for ans_ in ans_prob_dict:
            for j, _text in enumerate(texts):
                text_ = pre.prep_ans(_text)
                texts[j] = text_
                if ans_ == text_:
                    ans_score[self.ans_to_ix.__len__() + j] = pre.get_score(ans_prob_dict[ans_])
            if ans_ in self.ans_to_ix:
                ans_score[self.ans_to_ix[ans_]] = pre.get_score(ans_prob_dict[ans_])
        if text_feats.shape[0]:
            text_feats = np.pad(
                text_feats,
                ((0, 14 - text_feats.shape[0]), (0, 0)),
                mode='constant',
                constant_values=0
            )
        else:
            text_feats = np.zeros((14, 300))
        # cat = np.array([])
        cat = np.ones(self.ans_to_ix.__len__() + 14, np.bool)
        for i in self.types_dict[ans['answer_type']]:
            cat[i] = False
        for j in range(14):
            cat[j + self.ans_to_ix.__len__()] = False
        return img_feat_x, \
               torch.from_numpy(ques_ix), \
               torch.from_numpy(ans_score), idx, torch.from_numpy(cat), torch.from_numpy(text_feats.astype(np.float32)), texts
