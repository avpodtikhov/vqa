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
from torchvision.models import resnet152

class VQAv2(Dataset):
    def __init__(self):
        self.img_path_list = []
        for entry in os.listdir(my_cfg.VAL['Feats']):
            if os.path.isfile(os.path.join(my_cfg.VAL['Feats'], entry)):
                self.img_path_list.append(os.path.join(my_cfg.VAL['Feats'], entry))
        for entry in os.listdir(my_cfg.TRAIN['Feats']):
            if os.path.isfile(os.path.join(my_cfg.TRAIN['Feats'], entry)):
                self.img_path_list.append(os.path.join(my_cfg.TRAIN['Feats'], entry))
        '''
        self.qns_path_list = []
        for entry in os.listdir(my_cfg.VAL['Questions_feat']):
            if os.path.isfile(os.path.join(my_cfg.VAL['Questions_feat'], entry)):
                self.qns_path_list.append(os.path.join(my_cfg.VAL['Questions_feat'], entry))
        for entry in os.listdir(my_cfg.TRAIN['Questions_feat']):
            if os.path.isfile(os.path.join(my_cfg.TRAIN['Questions_feat'], entry)):
                self.qns_path_list.append(os.path.join(my_cfg.TRAIN['Questions_feat'], entry))
        '''
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
        
        '''
        self.id_to_ques_path = {}
        for qn in self.qns_path_list:
            self.id_to_ques_path[int(qn.split('/')[-1].split('.')[0])] = qn
        '''

        self.id_to_img_path = {}
        for im in self.img_path_list:
            self.id_to_img_path[int(im.split('/')[-1].split('_')[-1].split('.')[0])] = im

        self.ans_to_ix, self.ix_to_ans = json.load(open(my_cfg.answer_dict, 'r'))

        self.token_to_ix, self.pretrained_emb = pre.tokenize(self.qns_list)

        self.ans_size = len(self.ans_to_ix)
        self.token_size = len(self.token_to_ix)
        self.data_size = len(self.ans_list)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)
        
        ans = self.ans_list[idx]
        qid = int(ans['question_id'])
        ques = self.id_to_ques[qid]

        id = int(ans['image_id'])
        img_path = self.id_to_img_path[id]
        img_feat = np.load(img_path, allow_pickle=True)['arr_0'][()]

        img_feat_x = img_feat['x']
        boxes = img_feat['boxes']

        if img_feat_x.shape[0] > my_cfg.img_feat_pad_size:
            img_feat_x = img_feat_x[:my_cfg.img_feat_pad_size]
        
        img_feat_x = np.pad(
            img_feat_x,
            ((0, my_cfg.img_feat_pad_size - img_feat_x.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )
        ques_ix = np.zeros(my_cfg.max_token, np.int64)
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for ix, word in enumerate(words):
            if word in self.token_to_ix:
                ques_ix[ix] = self.token_to_ix[word]
            else:
                ques_ix[ix] = self.token_to_ix['UNK']
            if ix + 1 == my_cfg.max_token:
                break
        # Process answer
        ans_score = np.zeros(self.ans_to_ix.__len__(), np.float32)
        ans_prob_dict = {}

        for ans_ in ans['answers']:
            ans_proc = pre.prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1

        for ans_ in ans_prob_dict:
            if ans_ in self.ans_to_ix:
                ans_score[self.ans_to_ix[ans_]] = pre.get_score(ans_prob_dict[ans_])
        
        # np.save(my_cfg.TRAIN['ProcessedA'] + str(qid) + 'npy', ques_ix)
        return torch.from_numpy(img_feat_x), \
               torch.from_numpy(ques_ix), \
               torch.from_numpy(ans_score), idx