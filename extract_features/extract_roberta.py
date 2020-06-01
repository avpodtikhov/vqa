import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from tqdm import tqdm
import json

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('roberta-large')
bert = BertModel.from_pretrained('roberta-large')
bert.cuda()
bert.eval()

max_token = 36

with open('/mnt/data/users/apodtikhov/vqa/mcan/v2_OpenEnded_mscoco_train2014_questions.json') as json_file:
    qns_list = json.load(json_file)['questions']

id_to_ques = {}
for i in tqdm(range(len(qns_list))):
    ques = qns_list[i]
    marked_text = "[CLS] " + ques['question'] + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    if len(indexed_tokens) > max_token:
        sep = indexed_tokens[-1]
        indexed_tokens = indexed_tokens[:max_token]
        indexed_tokens[-1] = sep
    for i in range(len(indexed_tokens), max_token):
        indexed_tokens.append(0)
    segments_ids = [1] * max_token
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()
    with torch.no_grad():
        encoded_layers, _ = bert(tokens_tensor, segments_tensors)
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)
    token_vecs_sum = []
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec.cpu().numpy())
    token_vecs_sum = np.array(token_vecs_sum)
    np.save('/mnt/data/users/apodtikhov/vqa/roberta/train2014_qns/'+ str(ques['question_id']) + '.npy', token_vecs_sum)

with open('/mnt/data/users/apodtikhov/vqa/mcan/v2_OpenEnded_mscoco_val2014_questions.json') as json_file:
    qns_list = json.load(json_file)['questions']

id_to_ques = {}
for i in tqdm(range(len(qns_list))):
    ques = qns_list[i]
    marked_text = "[CLS] " + ques['question'] + " [SEP]"
    # Split the sentence into tokens.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    if len(indexed_tokens) > max_token:
        sep = indexed_tokens[-1]
        indexed_tokens = indexed_tokens[:max_token]
        indexed_tokens[-1] = sep
    for i in range(len(indexed_tokens), max_token):
        indexed_tokens.append(0)
    segments_ids = [1] * max_token
    tokens_tensor = torch.tensor([indexed_tokens]).cuda()
    segments_tensors = torch.tensor([segments_ids]).cuda()
    with torch.no_grad():
        encoded_layers, _ = bert(tokens_tensor, segments_tensors)
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)
    token_vecs_sum = []
    for token in token_embeddings:
        # `token` is a [12 x 768] tensor
        # Sum the vectors from the last four layers.
        sum_vec = torch.sum(token[-4:], dim=0)
        # Use `sum_vec` to represent `token`.
        token_vecs_sum.append(sum_vec.cpu().numpy())
    token_vecs_sum = np.array(token_vecs_sum)
    np.save('/mnt/data/users/apodtikhov/vqa/roberta/val2014_qns/'+ str(ques['question_id']) + '.npy', token_vecs_sum)
