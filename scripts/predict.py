import os.path as osp
import huepy as hue

import numpy as np
import torch
from torch.backends import cudnn

import sys
sys.path.append('./')
from configs import args_faster_rcnn_mae

from glob import glob
import matplotlib.pyplot as plt

from lib.datasets import get_data_loader
from lib.model.faster_rcnn_mae import get_mae_model
from lib.utils.misc import lazy_arg_parse, Nestedspace, \
    resume_from_checkpoint
from lib.utils.evaluator import inference, detection_performance_calc


def main(new_args, get_model_fn):

    args = Nestedspace()
    print(args)
    args.load_from_json(osp.join(new_args.path, 'args.json'))
    args.from_dict(new_args.to_dict())  # override previous args

    device = torch.device(args.device)
    cudnn.benchmark = False

    print(hue.info(hue.bold(hue.lightgreen(
        'Working directory: {}'.format(args.path)))))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gallery_loader, probe_loader = get_data_loader(args, train=False)

    model = get_model_fn(args, training=False,
                         pretrained_backbone=False)
    model.to(device)

    args.resume = osp.join(args.path, 'checkpoint.pth')
    args, model, _, _ = resume_from_checkpoint(args, model)

    name_to_boxes, all_feats, probe_feats = \
        inference(model, gallery_loader, probe_loader, device)

    # Get gallery images
    gallery_imgs_ = sorted(glob('./demo/frames/*.jpg'))
    gallery_imgs_ = gallery_imgs_[:len(gallery_imgs_)-1]
    gallery_imgs = [kk.split('/')[-1] for kk in gallery_imgs_]
    gallery_imgs = [kk.split('\\')[-1] for kk in gallery_imgs]
    i = 0
    for gallery_img in gallery_imgs:
        print('\n', gallery_img, '...',)
        boxes_ = name_to_boxes[gallery_img]
        boxes = boxes_[:, :-1]
        print(boxes)
        features = all_feats[i]
        if boxes is None:
            # print gallery_img, 'no detections'
            print(gallery_img, 'no detections')
            continue
        # Compute pairwise cosine similarities,
        #   equals to inner-products, as features are already L2-normed
        # similarities = features.dot(probe_feats)
        similarities = boxes_[:, -1]
        print(similarities)

        # Visualize the results
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.imshow(plt.imread(gallery_imgs_[i]))
        plt.axis('off')
        for box, sim in zip(boxes, similarities):
            if sim > 0.5:
                # x1, y1, x2, y2, _ = box
                x1, y1, x2, y2 = box
                ax.add_patch(
                    plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  fill=False, edgecolor='#4CAF50', linewidth=3.5))
                ax.add_patch(
                    plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  fill=False, edgecolor='white', linewidth=1))
                ax.text(x1 + 5, y1 - 18, '{:.3f}'.format(sim),
                        bbox=dict(facecolor='#4CAF50', linewidth=0),
                        fontsize=20, color='white')
            break
        plt.tight_layout()
        gallery_img = './demo/frames/' + gallery_img
        fig.savefig(gallery_img.replace('frames', 'result'))
        plt.show()
        plt.close(fig)
        i = i + 1

    return 0


if __name__ == '__main__':
    arg_parser = args_faster_rcnn_mae()
    new_args = lazy_arg_parse(arg_parser)
    print(new_args)
    print('\nUsing MAE test !')

    fn = get_mae_model

    main(new_args, fn)
