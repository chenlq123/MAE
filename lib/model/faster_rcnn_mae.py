from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import warnings
from torch.jit.annotations import Tuple, List, Dict, Optional

from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from .resnet_backbone import resnet_backbone

from ..loss import OIMLoss

from ..utils.transforms import get_transform
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
import numpy as np
import time

paths = {
    'CUHK-SYSU': r'G:\CLQ\Postgraduate\ReID_Project\NAE/data/CUHK-SYSU/CUHK-SYSU/Image/SSM_parse/',
    'PRW': r'G:/CLQ/Postgraduate\ReID_Project\NAE/data/PRW/frames_parse/'}
# test for our example
# paths = {
#     'CUHK-SYSU': r'G:\CLQ\Postgraduate\ReID_Project\NAE_parse6_5final/demo/frames_parse/',
#     'PRW': r'G:\CLQ\Postgraduate\ReID_Project\NAE_parse6_5final/demo/frames_parse/'}

class FasterRCNN_MAE(GeneralizedRCNN):
    """
    See https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py#L26
    """

    def __init__(self, backbone,
                 num_classes=None, num_pids=5532, num_cq_size=5000,
                 # transform parameters
                 min_size=900, max_size=1500,
                 image_mean=None, image_std=None,
                 # Anchor settings:
                 anchor_scales=None, anchor_ratios=None,
                 # RPN parameters
                 rpn_anchor_generator=None, rpn_head=None,
                 rpn_pre_nms_top_n_train=12000, rpn_pre_nms_top_n_test=6000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=300,
                 rpn_nms_thresh=0.7,
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
                 # Box parameters
                 rcnn_bbox_bn=True,
                 box_roi_pool=None, feat_head=None, box_predictor=None,
                 box_score_thresh=0.0, box_nms_thresh=0.4, box_detections_per_img=300,
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.1,
                 box_batch_size_per_image=128, box_positive_fraction=0.5,
                 bbox_reg_weights=None,
                 # ReID parameters
                 embedding_head=None, reid_loss=None,
                 args=None, fpost=None):
        self.args = args

        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                'backbone should contain an attribute out_channels '
                'specifying the number of output channels (assumed to be the '
                'same for all the levels)')
        assert isinstance(rpn_anchor_generator, (AnchorGenerator, type(None)))
        assert isinstance(box_roi_pool, (MultiScaleRoIAlign, type(None)))

        if num_classes is not None:
            if box_predictor is not None:
                raise ValueError(
                    'num_classes should be None when box_predictor is specified')
        else:
            if box_predictor is None:
                raise ValueError('num_classes should not be None when box_predictor'
                                 'is not specified')

        out_channels = backbone.out_channels

        if rpn_anchor_generator is None:
            if anchor_scales is None:
                anchor_scales = ((32, 64, 128, 256, 512),)
            if anchor_ratios is None:
                anchor_ratios = ((0.5, 1.0, 2.0),)
            rpn_anchor_generator = AnchorGenerator(
                anchor_scales, anchor_ratios)
        if rpn_head is None:
            rpn_head = RPNHead(
                out_channels,
                rpn_anchor_generator.num_anchors_per_location()[0])

        rpn_pre_nms_top_n = dict(
            training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(
            training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)

        rpn = self._set_rpn(
            rpn_anchor_generator, rpn_head,
            rpn_fg_iou_thresh, rpn_bg_iou_thresh,
            rpn_batch_size_per_image, rpn_positive_fraction,
            rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        if box_roi_pool is None:
            box_roi_pool = MultiScaleRoIAlign(
                featmap_names=['feat_res4'],
                output_size=14,
                sampling_ratio=2)

        if feat_head is None:
            raise ValueError('feat_head should be specified manually.')

        if box_predictor is None:
            box_predictor = CoordRegressor(
                2048, num_classes,
                rcnn_bbox_bn)

        if embedding_head is None:
            embedding_head = MAE_EmbeddingProj(
                featmap_names=['feat_res4', 'feat_res5'],
                in_channels=[1024, 2048],
                dim=256)

        if reid_loss is None:
            reid_loss = OIMLoss(
                384, num_pids, num_cq_size,
                0.5, 30.0)
        roi_heads = self._set_roi_heads(
            embedding_head, reid_loss, fpost,
            box_roi_pool, feat_head, box_predictor,
            box_fg_iou_thresh, box_bg_iou_thresh,
            box_batch_size_per_image, box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh, box_nms_thresh, box_detections_per_img)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]


        transform = GeneralizedRCNNTransform(
            min_size, max_size, image_mean, image_std)

        super(FasterRCNN_MAE, self).__init__(
            backbone, rpn, roi_heads, transform)
    def forward(self, images, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # --------------------------------
        targets_parse = targets
        if not self.training:
            targets = None
        # --------------------------------

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenrate box
                    bb_idx = degenerate_boxes.any(dim=1).nonzero().view(-1)[0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invaid box {} for target at index {}."
                                     .format(degen_bb, target_idx))
        features = self.backbone(images.tensors)

        img_parse = []
        for target in targets_parse:
            img_parse_path = paths[self.args.dataset] + target['im_name'] + '.mat'
            img_parse_ = loadmat(img_parse_path)
            img_parse_ = img_parse_['img_parse']
            if target["flipped"] and self.training:
                img_parse_ = np.fliplr(img_parse_)

            img_parse_ = torch.from_numpy(img_parse_.copy())
            img_parse_ = img_parse_.cuda()
            img_parse.append(img_parse_)

        # 5 attribute
        all_parse, h_parse, u_parse, d_parse, s_parse, b_parse = self.split_parse(img_parse)

        # resize
        all_parse = self.transform_parse(all_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        h_parse = self.transform_parse(h_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        u_parse = self.transform_parse(u_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        d_parse = self.transform_parse(d_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        s_parse = self.transform_parse(s_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        b_parse = self.transform_parse(b_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])

        mask_parse = OrderedDict([('head', h_parse), ('up', u_parse), ('down', d_parse),
                                  ('shoes', s_parse), ('bag', b_parse), ('all', all_parse)])

        if isinstance(features, torch.Tensor):  # 未执行
            features = OrderedDict([('0', features)])

        proposals, proposal_losses = self.rpn(images, features, targets)
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets, mask_parse)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)

    def transform_parse(self, images, img_size, img_final_size, f_size):
        for i in range(len(images)):
            image = images[i]
            image = image.float()
            if image.dim() != 2:
                raise ValueError("images is expected to be a list of 2d tensors "
                                 "of shape [H, W], got {}".format(image.shape))
            image = image.unsqueeze(0).unsqueeze(0)
            image = torch.nn.functional.interpolate(image, size=list(img_size[i]))
            # aaa, bbb = torch.sort(image.squeeze().view(1, -1), descending=True)
            image = torch.div(image, 255.0)
            # aaa, bbb = torch.sort(image.squeeze().view(1, -1), descending=True)
            images[i] = image.squeeze(0)

        img_final_size = [len(images)] + [1] + list(img_final_size)
        batched_imgs = images[0].new_full(img_final_size, 0)
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        batched_imgs = torch.nn.functional.interpolate(batched_imgs, size=[f_size[0], f_size[1]])
        return batched_imgs

    def split_parse(self, images):
        head_parse = images.copy()
        u_parse = images.copy()
        d_parse = images.copy()
        s_parse = images.copy()
        bag_parse = images.copy()
        all_parse = images.copy()
        for i in range(len(images)):
            image = images[i]
            if image.dim() != 2:
                raise ValueError("images is expected to be a list of 2d tensors "
                                 "of shape [H, W], got {}".format(image.shape))
            zeros_ = torch.zeros_like(image)
            ones_ = torch.ones_like(image)
            ones255_ = torch.mul(ones_, 255)
            head_parse[i] = torch.where(image == 51, ones255_, zeros_)
            u_parse[i] = torch.where(image == 102, ones255_, zeros_)
            d_parse[i] = torch.where(image == 153, ones255_, zeros_)
            s_parse[i] = torch.where(image == 204, ones255_, zeros_)
            bag_parse[i] = torch.where(image == 255, ones255_, zeros_)
            all_parse[i] = torch.where(image > 0, ones255_, zeros_)

        return all_parse, head_parse, u_parse, d_parse, s_parse, bag_parse

    def _set_rpn(self, *args):
        return RegionProposalNetwork(*args)

    def _set_roi_heads(self, *args):
        return MAERoiHeads(*args)

    def ex_feat(self, images, targets, mode='det'):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result: (tuple(Tensor)): list of 1 x d embedding for the RoI of each image

        """
        if mode == 'det':
            return self.ex_feat_by_roi_pooling(images, targets)
        elif mode == 'reid':
            return self.ex_feat_by_img_crop(images, targets)

    def ex_feat_by_roi_pooling(self, images, targets):
        targets_parse = targets
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        proposals = [x['boxes'] for x in targets]
        img_parse = []
        for target in targets_parse:
            img_parse_path = paths[self.args.dataset] + target['im_name'] + '.mat'
            img_parse_ = loadmat(img_parse_path)
            img_parse_ = img_parse_['img_parse']
            img_parse_ = torch.from_numpy(img_parse_.copy())
            img_parse_ = img_parse_.cuda()
            img_parse.append(img_parse_)

        all_parse, h_parse, u_parse, d_parse, s_parse, b_parse = self.split_parse(img_parse)

        all_parse = self.transform_parse(all_parse, images.image_sizes, images.tensors.shape[-2:],
                                         features['feat_res4'].shape[-2:])
        h_parse = self.transform_parse(h_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        u_parse = self.transform_parse(u_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        d_parse = self.transform_parse(d_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        s_parse = self.transform_parse(s_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])
        b_parse = self.transform_parse(b_parse, images.image_sizes, images.tensors.shape[-2:],
                                       features['feat_res4'].shape[-2:])

        mask_parse = OrderedDict([('head', h_parse), ('up', u_parse), ('down', d_parse),
                                  ('shoes', s_parse), ('bag', b_parse), ('all', all_parse)])

        head_features = {}
        u_features = {}
        d_features = {}
        s_features = {}
        bag_features = {}
        featuremask = torch.mul(features['feat_res4'], mask_parse['all'])
        featuremask = self.roi_heads.featpostp(featuremask)
        head_features['feat_res4'] = torch.mul(featuremask, mask_parse['head'])
        u_features['feat_res4'] = torch.mul(featuremask, mask_parse['up'])
        d_features['feat_res4'] = torch.mul(featuremask, mask_parse['down'])
        s_features['feat_res4'] = torch.mul(featuremask, mask_parse['shoes'])
        bag_features['feat_res4'] = torch.mul(featuremask, mask_parse['bag'])

        roi_head_features = self.roi_heads.box_roi_pool(head_features, proposals, images.image_sizes)
        roi_up_features = self.roi_heads.box_roi_pool(u_features, proposals, images.image_sizes)
        roi_down_features = self.roi_heads.box_roi_pool(d_features, proposals, images.image_sizes)
        roi_shoes_features = self.roi_heads.box_roi_pool(s_features, proposals, images.image_sizes)
        roi_bag_features = self.roi_heads.box_roi_pool(bag_features, proposals, images.image_sizes)

        mask_parse_cat = torch.cat((roi_head_features, roi_up_features, roi_down_features,
                                    roi_shoes_features, roi_bag_features), 1)

        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])


        roi_pooled_features = self.roi_heads.box_roi_pool(
            features, proposals, images.image_sizes)
        rcnn_features = self.roi_heads.feat_head(roi_pooled_features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])
        embeddings, norms = self.roi_heads.embedding_head(rcnn_features, mask_parse_cat)  # 这里我也有改动
        return embeddings.split(1, 0)

    def ex_feat_by_img_crop(self, images, targets):
        assert len(images) == 1, 'Only support batch_size 1 in this mode'

        images, targets = self.transform(images, targets)
        x1, y1, x2, y2 = map(lambda x: int(round(x)),
                             targets[0]['boxes'][0].tolist())
        input_tensor = images.tensors[:, :, y1:y2 + 1, x1:x2 + 1]
        features = self.backbone(input_tensor)
        features = features.values()[0]
        rcnn_features = self.roi_heads.feat_head(features)
        if isinstance(rcnn_features, torch.Tensor):
            rcnn_features = OrderedDict([('feat_res5', rcnn_features)])
        embeddings, norms = self.roi_heads.embedding_head(rcnn_features)
        return embeddings.split(1, 0)


class MAERoiHeads(RoIHeads):
    def __init__(self, embedding_head, reid_loss, fpost, *args, **kwargs):
        super(MAERoiHeads, self).__init__(*args, **kwargs)
        self.embedding_head = embedding_head
        self.reid_loss = reid_loss
        self.featpostp = fpost

    @property
    def feat_head(self):  # re-name
        return self.box_head



    def forward(self, features, proposals, image_shapes, targets=None, mask_parse=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, \
                    'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, \
                    'target labels must of int64 type'

        if self.training:

            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)

        roi_pooled_features = self.box_roi_pool(features, proposals, image_shapes)

        head_features = {}
        u_features = {}
        d_features = {}
        s_features = {}
        bag_features = {}
        featuremask = torch.mul(features['feat_res4'], mask_parse['all'])
        featuremask = self.featpostp(featuremask)
        head_features['feat_res4'] = torch.mul(featuremask, mask_parse['head'])
        u_features['feat_res4'] = torch.mul(featuremask, mask_parse['up'])
        d_features['feat_res4'] = torch.mul(featuremask, mask_parse['down'])
        s_features['feat_res4'] = torch.mul(featuremask, mask_parse['shoes'])
        bag_features['feat_res4'] = torch.mul(featuremask, mask_parse['bag'])

        roi_head_features = self.box_roi_pool(head_features, proposals, image_shapes)
        roi_up_features = self.box_roi_pool(u_features, proposals, image_shapes)
        roi_down_features = self.box_roi_pool(d_features, proposals, image_shapes)
        roi_shoes_features = self.box_roi_pool(s_features, proposals, image_shapes)
        roi_bag_features = self.box_roi_pool(bag_features, proposals, image_shapes)


        mask_parse_cat = torch.cat((roi_head_features, roi_up_features, roi_down_features,
                                    roi_shoes_features, roi_bag_features), 1)


        rcnn_features = self.feat_head(roi_pooled_features)
        box_regression = self.box_predictor(rcnn_features['feat_res5'])
        embeddings_, class_logits = self.embedding_head(rcnn_features, mask_parse_cat)

        result, losses = [], {}
        if self.training:
            det_labels = [y.clamp(0, 1) for y in labels]
            loss_detection, loss_box_reg = \
                mae_rcnn_loss(class_logits, box_regression,
                                     det_labels, regression_targets)
            loss_reid = self.reid_loss(embeddings_, labels)

            losses = dict(loss_detection=loss_detection,
                          loss_box_reg=loss_box_reg,
                          loss_reid=loss_reid)
        else:
            boxes, scores, embeddings, labels = \
                self.postprocess_detections(class_logits, box_regression, embeddings_, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings[i],
                    )
                )
        # Mask and Keypoint losses are deleted
        return result, losses

    def postprocess_detections(self, class_logits, box_regression, embeddings_, proposals, image_shapes):
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = torch.sigmoid(class_logits)
        embeddings_ = embeddings_ * pred_scores.view(-1, 1)  # CWS

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings_.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(pred_boxes, pred_scores, pred_embeddings, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)
            # embeddings are already personized.

            # batch everything, by making every class prediction be a separate
            # instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            # embeddings = embeddings.reshape(-1, self.embedding_head.dim)
            embeddings = embeddings.reshape(-1, self.reid_loss.num_features)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = boxes[
                inds], scores[inds], labels[inds], embeddings[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels, embeddings = boxes[keep], scores[keep], \
                labels[keep], embeddings[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels

class featpostprocess(nn.Module):
    def __init__(self):
        super(featpostprocess, self).__init__()
        self.fprocess = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1024, 512, kernel_size=1, stride=1, bias=False)),
            ('conv2', nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv2d(256, 154, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1)),
            ('bn2', nn.BatchNorm2d(154)),
            ('relu2', nn.ReLU(inplace=True)),
        ]))

    def forward(self, x):
        return self.fprocess(x)


class MAE_EmbeddingProj(nn.Module):

    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256):
        super(MAE_EmbeddingProj, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = list(map(int, in_channels))
        self.dim = int(dim)

        self.feat_parse1 = nn.Sequential(
            nn.Conv2d(770, 770, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(770, 770, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1),
            nn.Conv2d(770, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.feat_parse2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, groups=1, bias=False, dilation=1),
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512))
        self.feat_parse3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, stride=1, bias=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0, groups=1, bias=False, dilation=1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.proj_parse = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128))
        init.normal_(self.proj_parse[0].weight, std=0.01)
        init.normal_(self.proj_parse[1].weight, std=0.01)
        init.constant_(self.proj_parse[0].bias, 0)
        init.constant_(self.proj_parse[1].bias, 0)
        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()  #  return [128，128]
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            proj = nn.Sequential(
                nn.Linear(in_chennel, indv_dim),
                nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps, featparse):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        '''
        fparse = self.feat_parse1(featparse)
        fparse = fparse + self.feat_parse2(fparse)
        fparse = self.feat_parse3(fparse)
        fparse = F.adaptive_max_pool2d(fparse, 1)
        fparse = fparse.flatten(start_dim=1)
        fparse = self.proj_parse(fparse)

        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)  # [m,n,1,1] to [m,n]
                outputs.append(
                    self.projectors[k](v)
                )
            embeddings = torch.cat(outputs, dim=1)
            embeddings = torch.cat((embeddings, fparse), dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    @property
    def rescaler_weight(self):
        return self.rescaler.weight.item()

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp


class CoordRegressor(nn.Module):
    """
    bounding box regression layers, without classification layer.
    for Fast R-CNN.
    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
                           default = 2 for pedestrian detection，为什么对于行人检测就是2呢
    """

    def __init__(self, in_channels, num_classes=2, RCNN_bbox_bn=True):
        super(CoordRegressor, self).__init__()
        if RCNN_bbox_bn:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes),
                nn.BatchNorm1d(4 * num_classes))
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)
        self.cls_score = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas

# box regression和person/bg classification loss
def mae_rcnn_loss(class_logits, box_regression, labels, regression_targets):
    """
    Computes the loss for MAE R-CNN.
    Arguments:
        class_logits (Tensor), size = (N, )
        box_regression (Tensor)
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.binary_cross_entropy_with_logits(
        class_logits, labels.float())

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
    labels_pos = labels[sampled_pos_inds_subset]
    N = class_logits.size(0)
    box_regression = box_regression.reshape(N, -1, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss

# construct MAE model
def get_mae_model(args, training=True, pretrained_backbone=True):
    backbone, conv_head = resnet_backbone('resnet50', pretrained_backbone)
    coord_fc = CoordRegressor(2048, num_classes=2,
                              RCNN_bbox_bn=args.rcnn_bbox_bn)
    embedding_head = MAE_EmbeddingProj(
        featmap_names=['feat_res4', 'feat_res5'],
        in_channels=[1024, 2048],
        dim=256)
    featpost = featpostprocess()
    phase_args = args.train if training else args.test
    model = FasterRCNN_MAE(backbone,
                                 feat_head=conv_head,
                                 box_predictor=coord_fc,
                                 embedding_head=embedding_head,
                                 num_pids=args.num_pids, num_cq_size=args.num_cq_size,
                                 min_size=phase_args.min_size, max_size=phase_args.max_size,
                                 anchor_scales=(tuple(args.anchor_scales),),
                                 anchor_ratios=(tuple(args.anchor_ratios),),
                                 # RPN parameters
                                 rpn_pre_nms_top_n_train=args.train.rpn_pre_nms_top_n,
                                 rpn_post_nms_top_n_train=args.train.rpn_post_nms_top_n,
                                 rpn_pre_nms_top_n_test=args.test.rpn_pre_nms_top_n,
                                 rpn_post_nms_top_n_test=args.test.rpn_post_nms_top_n,
                                 rpn_nms_thresh=phase_args.rpn_nms_thresh,
                                 rpn_fg_iou_thresh=args.train.rpn_positive_overlap,
                                 rpn_bg_iou_thresh=args.train.rpn_negative_overlap,
                                 rpn_batch_size_per_image=args.train.rpn_batch_size,
                                 rpn_positive_fraction=args.train.rpn_fg_fraction,
                                 # Box parameters
                                 rcnn_bbox_bn=args.rcnn_bbox_bn,
                                 box_score_thresh=args.train.fg_thresh,
                                 box_nms_thresh=args.test.nms,  # inference only
                                 box_detections_per_img=phase_args.rpn_post_nms_top_n,  # use all
                                 box_fg_iou_thresh=args.train.bg_thresh_hi,
                                 box_bg_iou_thresh=args.train.bg_thresh_lo,
                                 box_batch_size_per_image=args.train.rcnn_batch_size,
                                 box_positive_fraction=args.train.fg_fraction,  # for proposals
                                 bbox_reg_weights=args.train.box_regression_weights,
                                 args=args, fpost=featpost
                                 )

    if training:
        model.train()
    else:
        model.eval()

    return model
