import os

VQA_FOLDER = '/mnt/data/users/apodtikhov/vqa/'

data = {'IMAGES' : os.path.join(VQA_FOLDER, 'images'),
        'BOTTOM-UP' : os.path.join(VQA_FOLDER, 'mcan'),
        'CASCADE' : os.path.join(VQA_FOLDER, 'cascade'),
        'PANOPTIC' : os.path.join(VQA_FOLDER, 'panoptic'),
        'PYTHIA' : os.path.join(VQA_FOLDER, 'boxes'),
        'BERT' : os.path.join(VQA_FOLDER, 'bert'),
        'ROBERTA' : os.path.join(VQA_FOLDER, 'roberta'),
        'ALBERT' : os.path.join(VQA_FOLDER, 'albert'),
        'OCR' : os.path.join(VQA_FOLDER, 'extracted_texts'),
        'QUESTIONS' : VQA_FOLDER,
        'ANNOTATIONS' : VQA_FOLDER}

answer_dict = VQA_FOLDER + 'answer_dict.json'