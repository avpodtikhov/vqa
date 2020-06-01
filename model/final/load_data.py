import json
import numpy as np
import os
from PIL import Image
import final.preprocessing as pre
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import re
import configs.default_paths as paths

class VQAv2(Dataset):
    def __init__(self, cfg, mode):
        self.mode = mode
        self.cfg = cfg
        self.img_path_list = []
        self.qns_path_list = []
        self.qns_list = []
        self.ans_list = []
        self.id_to_ques_path = {}
        self.id_to_img_path = {}

        images_path = os.path.join(cfg['IMAGE_MODEL'], self.mode + '2014')
        for entry in os.listdir(images_path):
            if os.path.isfile(os.path.join(images_path, entry)):
                _path = os.path.join(images_path, entry)
                self.img_path_list.append(_path)
                self.id_to_img_path[int(_path.split('/')[-1].split('_')[-1].split('.')[0])] = _path
        self.pretrained_emb = []
        qns_path = os.path.join(cfg['QUES_MODEL'], self.mode + '2014_qns')
        for entry in os.listdir(qns_path):
            if os.path.isfile(os.path.join(qns_path, entry)):
                _path = os.path.join(qns_path, entry)
                self.qns_path_list.append(_path)
                self.id_to_ques_path[int(_path.split('/')[-1].split('.')[0])] = _path
        with open(os.path.join(paths.data['QUESTIONS'], 'v2_OpenEnded_mscoco_' + self.mode + '2014_questions.json')) as json_file:
            self.qns_list += json.load(json_file)['questions']

        with open(os.path.join(paths.data['QUESTIONS'], 'v2_mscoco_' + self.mode + '2014_annotations.json')) as json_file:
            self.ans_list += json.load(json_file)['annotations']

        self.id_to_ques = {}
        for qn in self.qns_list:
            self.id_to_ques[int(qn['question_id'])] = qn
        
        self.token_to_ix, self.pretrained_emb = pre.tokenize(self.qns_list)
        self.ans_to_ix, self.ix_to_ans = json.load(open(paths.answer_dict, 'r'))

        with open('./final/indexes.json') as f:
            self.types_dict = json.load(f)

        self.ans_size = len(self.ans_to_ix)
        self.token_size = len(self.token_to_ix)
        self.data_size = len(self.ans_list)

    def __len__(self):
        return self.data_size

    def prep_read(self, id):
        img_path = self.id_to_img_path[id]
        texts = []
        if self.cfg['num_to_read']:
            if 'val' in img_path:
                texts_path = self.cfg['READ'] + 'val2014_text/' + img_path.split('/')[-1].split('.')[0] + '.jpg'
                feats_path = self.cfg['READ'] + 'val2014_text_1/' + img_path.split('/')[-1].split('.')[0] + '.jpg'
            else:
                texts_path = self.cfg['READ'] + 'train2014_text/' + img_path.split('/')[-1].split('.')[0] + '.jpg'
                feats_path = self.cfg['READ'] + 'train2014_text_1/' + img_path.split('/')[-1].split('.')[0] + '.jpg'
            texts = list(np.load(texts_path + '.npy'))
            text_feats = np.load(feats_path + '.npy')

            if len(texts) > self.cfg['num_to_read']:
                texts = texts[:self.cfg['num_to_read']]
        return texts, text_feats

    def __getitem__(self, idx):
        ans = self.ans_list[idx]
    
        qid = int(ans['question_id'])
        ques_path = self.id_to_ques_path[qid]
        ques_ix = np.load(ques_path)

        if ques_ix.shape[0] > self.cfg['max_token']:
            sep = ques_ix[-1]
            ques_ix = ques_ix[:self.cfg['max_token']]
            ques_ix[-1] = sep 

        ques_ix = np.pad(
            ques_ix,
            ((0, self.cfg['max_token'] - ques_ix.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        id = int(ans['image_id'])
        img_path = self.id_to_img_path[id]
        img_feat = np.load(img_path)
        boxes = img_feat['boxes']
        img_feat_x = img_feat['x']

        if img_feat_x.shape[0] > self.cfg['img_feat_pad_size']:
            img_feat_x = img_feat_x[:self.cfg['img_feat_pad_size']]

        img_feat_x = np.pad(
            img_feat_x,
            ((0, self.cfg['img_feat_pad_size'] - img_feat_x.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        boxes = np.pad(
            boxes,
            ((0, self.cfg['img_feat_pad_size'] - boxes.shape[0]), (0, 0)),
            mode='constant',
            constant_values=0
        )

        texts, text_feats = self.prep_read(id)

        # Process answer
        ans_score = np.zeros(self.ans_to_ix.__len__() + self.cfg['num_to_read'], np.float32)
        ans_prob_dict = {}

        for ans_ in ans['answers']:
            ans_proc = pre.prep_ans(ans_['answer'])
            if ans_proc not in ans_prob_dict:
                ans_prob_dict[ans_proc] = 1
            else:
                ans_prob_dict[ans_proc] += 1
        
        while len(texts) < self.cfg['num_to_read']:
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
        for j in range(self.cfg['num_to_read']):
            cat[j + self.ans_to_ix.__len__()] = False
        
        return torch.from_numpy(img_feat_x), \
               torch.from_numpy(ques_ix), \
               torch.from_numpy(ans_score), \
               idx, torch.from_numpy(cat), \
               torch.from_numpy(text_feats.astype(np.float32)), \
               texts, \
               torch.from_numpy(boxes).permute(1, 0)