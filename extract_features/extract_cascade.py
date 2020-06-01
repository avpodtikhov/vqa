import logging
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

from torchvision.ops import nms

from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs

with torch.cuda.device(2):

    def fast_rcnn_inference(boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image, box_features):
        result_per_image = [
            fast_rcnn_inference_single_image(
                boxes_per_image, scores_per_image, image_shape, score_thresh, nms_thresh, topk_per_image, box_features
            )
            for scores_per_image, boxes_per_image, image_shape in zip(scores, boxes, image_shapes)
        ]
        return [x[0] for x in result_per_image], [x[1] for x in result_per_image], [x[2] for x in result_per_image]

    from torchvision.ops import boxes as box_ops

    def fast_rcnn_inference_single_image(boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image, box_features):
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            box_features = box_features[valid_mask]

        scores = scores[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = Boxes(boxes.reshape(-1, 4))
        boxes.clip(image_shape)
        boxes = boxes.tensor.view(-1, 4)  # R x C x 4
        max_conf = torch.zeros((boxes.shape[0])).cuda()
        for cls_ind in range(0,scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            # dets = torch.cat([boxes, cls_scores.view(-1, 1)], 1)
            keep = nms(boxes, cls_scores, 0.3)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])
        keep_boxes = torch.where(max_conf >= 0.2)[0]
        if len(keep_boxes) < 36:
            keep_boxes = torch.argsort(max_conf, descending=True)[:36]
        elif len(keep_boxes) > 36:
            keep_boxes = keep_boxes[:36]
        boxes, scores = boxes[keep_boxes], scores[keep_boxes]
        box_features = box_features[keep_boxes] 
        result = Instances(image_shape)
        result.pred_boxes = Boxes(boxes)
        result.scores = scores
        result.pred_classes = keep_boxes
        return result, keep_boxes, box_features

    import cv2
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    import detectron2.data.transforms as T
    from detectron2.structures import ImageList
    # You may need to restart your runtime prior to this, to let your installation take effect
    # Some basic setup:
    # Setup detectron2 logger
    import detectron2
    from detectron2.utils.logger import setup_logger
    setup_logger()

    # import some common libraries
    import numpy as np
    import cv2
    import random

    # import some common detectron2 utilities
    from detectron2 import model_zoo
    from detectron2.engine import DefaultPredictor, DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from torch.autograd.function import Function

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")

    image_model = build_model(cfg)

    image_model.cuda()
    image_model.eval()
    checkpointer = DetectionCheckpointer(image_model)
    checkpointer.load(cfg.MODEL.WEIGHTS)


    class _ScaleGradient(Function):
        @staticmethod
        def forward(ctx, input, scale):
            ctx.scale = scale
            return input

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output * ctx.scale, None


    def get_image_features(img_paths, folder):
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            inputs = []
            for i in range(len(img_paths)):
                im = cv2.imread(img_paths[i])
                original_image = im

                transform_gen = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
                # inputs = {"image": image, "height": height, "width": width}
                # predictions = model.backbone(torch.tensor([image]))

                height, width = original_image.shape[:2]
                image = transform_gen.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

                inputs.append({"image": image, "height": height, "width": width})

            images = image_model.preprocess_image(inputs)

            predictions = image_model.backbone(images.tensor.cuda())
            proposals, _ = image_model.proposal_generator(images, predictions, None)
            # pred_instances = model.roi_heads(123123, 12412, 14212, None) #
            features = [predictions[f] for f in image_model.roi_heads.in_features]
            head_outputs = []
            image_sizes = [x.image_size for x in proposals]
            for k in range(image_model.roi_heads.num_cascade_stages):
                if k > 0:
                    # The output boxes of the previous stage are the input proposals of the next stage
                    proposals = image_model.roi_heads._create_proposals_from_boxes(
                        head_outputs[-1].predict_boxes(), image_sizes
                    )
                box_features = image_model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
                box_features = _ScaleGradient.apply(box_features, 1.0 / image_model.roi_heads.num_cascade_stages)
                box_features = image_model.roi_heads.box_head[k](box_features)
                pred_class_logits, pred_proposal_deltas = image_model.roi_heads.box_predictor[k](box_features)
                outputs = FastRCNNOutputs(
                    image_model.roi_heads.box2box_transform[k],
                    pred_class_logits,
                    pred_proposal_deltas,
                    proposals,
                    image_model.roi_heads.smooth_l1_beta,
                )
                head_outputs.append(outputs)
            scores_per_stage = [h.predict_probs() for h in head_outputs]
            scores = [
                            sum(list(scores_per_image)) * (1.0 / image_model.roi_heads.num_cascade_stages)
                            for scores_per_image in zip(*scores_per_stage)
                    ]    
            boxes = head_outputs[-1].predict_boxes()
            pred_instances = fast_rcnn_inference(boxes, scores, image_sizes, image_model.roi_heads.test_score_thresh, image_model.roi_heads.test_nms_thresh, image_model.roi_heads.test_detections_per_img, box_features)
            outputs = GeneralizedRCNN._postprocess(pred_instances[0], inputs, images.image_sizes), pred_instances[2]
        for path, instance, feats in zip(img_paths, outputs[0], outputs[1]):
            d = {}
            d['boxes'] = instance['instances'].pred_boxes.tensor.cpu().numpy()
            d['x'] = feats.cpu().numpy()
            np.savez_compressed(folder + path.split('/')[-1] + '.npz', d)
            # return d['x'].shape[0]
    from config_cascade import *
    import os 
    img_path_list = []
    for entry in os.listdir(TRAIN['Images']):
        if os.path.isfile(os.path.join(TRAIN['Images'], entry)):
            img_path_list.append(os.path.join(TRAIN['Images'], entry))
    from tqdm import tqdm
    for i in tqdm(range(len(img_path_list))):
        pth = img_path_list[i]
        if not os.path.exists(pth):
            print(pth)
    '''
    num_features_total = 0.
    for i in tqdm(range(8, len(img_path_list) + 1, 8)):
        d = get_image_features(img_path_list[i - 8 : i], VAL['Feats'])
        # num_features_total += d
        if i % 1000 == 0:
            print(num_features_total / i)
    get_image_features(img_path_list[i: ], VAL['Feats'])
    from config import *
    import os 
    img_path_list = []
    for entry in os.listdir(TRAIN['Images']):
        if os.path.isfile(os.path.join(TRAIN['Images'], entry)):
            img_path_list.append(os.path.join(TRAIN['Images'], entry))
    from tqdm import tqdm 
    num_features_total = 0.
    for i in tqdm(range(16, len(img_path_list) + 1, 16)):
        d = get_image_features(img_path_list[i - 16 : i], TRAIN['Feats'])
        # num_features_total += d
        if i % 1000 == 0:
            print(num_features_total / i)
    get_image_features(img_path_list[i: ], TRAIN['Feats'])
'''
