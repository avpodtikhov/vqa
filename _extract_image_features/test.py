import detectron2
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os 
from tqdm import tqdm
import numpy as np
import torch

with torch.cuda.device(1):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    predictor = DefaultPredictor(cfg)

    av = 0.
    m = 0.
    img_path_list = []
    for entry in os.listdir('/mnt/data/users/apodtikhov/vqa/train2014/'):
        if os.path.isfile(os.path.join('/mnt/data/users/apodtikhov/vqa/train2014/', entry)):
            img_path_list.append(os.path.join('/mnt/data/users/apodtikhov/vqa/train2014/', entry))
            
    for i in tqdm(range(len(img_path_list))):
        im_name = img_path_list[i]
        im = cv2.imread(im_name)
        outputs = predictor(im)
        panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
        mask = panoptic_seg.cpu().numpy()
        np.save('/mnt/data/users/apodtikhov/vqa/train2014_mask/' + img_path_list[i].split('/')[-1] + '.npy', (len(segments_info), mask))
        av += len(segments_info)
        m = max(m, len(segments_info))

    print('Maximum', m)
    print('Average', av / (i + 1))