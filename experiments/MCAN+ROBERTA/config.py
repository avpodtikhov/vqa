# Paths
VQA_FOLDER = '/mnt/data/users/apodtikhov/vqa/'

TRAIN = {'Images' : VQA_FOLDER + 'mcan/train2014/',
         'Questions_feat' : VQA_FOLDER + 'roberta/train2014_qns/',
         'Questions' : VQA_FOLDER + 'mcan/v2_OpenEnded_mscoco_train2014_questions.json',
         'Answers' : VQA_FOLDER + 'mcan/v2_mscoco_train2014_annotations.json',
}

VAL = {'Images' : VQA_FOLDER + 'mcan/val2014/',
       'Questions_feat' : VQA_FOLDER + 'roberta/val2014_qns/',
       'Questions' : VQA_FOLDER + 'mcan/v2_OpenEnded_mscoco_val2014_questions.json',
       'Answers' : VQA_FOLDER + 'mcan/v2_mscoco_val2014_annotations.json',
}

#было log_bert_2
# model_log = VQA_FOLDER + 'models/mcan+bert/log.json'
# model_pth = VQA_FOLDER + 'models//mcan+bert/model.pth'
model_log = VQA_FOLDER + 'models/mcan+roberta/log.json'
model_pth = VQA_FOLDER + 'models/mcan+roberta/model_bert_2.pth'
answer_dict = VQA_FOLDER + 'answer_dict.json'

# Constants

## Prepare images, questions, answers

img_feat_pad_size = 100
max_token = 36 # 36

##  Hyperparameters for network

hidden_size = 512
glove_size = 300
img_feat_size = 2048
num_layers = 6
dropout_rate = 0.1
fc_size = 4 * hidden_size
multi_head = 8
hidden_size_head = int(hidden_size / multi_head)
flat_out_size = 1024
flat_mlp_size = 512

## Train parameters

LR_BASE = 0.0001
OPT_BETAS = (0.9, 0.98)
OPT_EPS = 1e-9
BATCH_SIZE=4
MAX_EPOCH=13
LR_DECAY_LIST = [10, 12]
LR_DECAY_R = 0.2
