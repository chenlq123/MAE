import os.path as osp
import huepy as hue
import socket
from datetime import datetime

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from ignite.engine.engine import Events

import sys
sys.path.append('./')
from configs import args_faster_rcnn_mae

from lib.model.faster_rcnn_mae import get_mae_model
from lib.datasets import get_data_loader
from lib.utils.distributed import init_distributed_mode, is_main_process
from lib.utils.misc import Nestedspace, get_optimizer, get_lr_scheduler
from lib.utils.trainer import get_trainer
from lib.utils.serialization import mkdir_if_missing


def main(args, get_model_fn):
    if args.distributed:
        init_distributed_mode(args)
    device = torch.device(args.device)
    cudnn.benchmark = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if is_main_process():
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        args.path = osp.join(
            args.path, current_time + '_' + socket.gethostname())
        mkdir_if_missing(args.path)
        print(hue.info(hue.bold(hue.lightgreen(
            'Working directory: {}'.format(args.path)))))
        if args.train.use_tfboard:
            tfboard = SummaryWriter(log_dir=args.path)
        args.export_to_json(osp.join(args.path, 'args.json'))
    else:
        tfboard = None
    # load dataset
    train_loader = get_data_loader(args, train=True)
    model = get_model_fn(args, training=True,
                         pretrained_backbone=True)
    model.to(device)

    optimizer = get_optimizer(args, model)
    lr_scheduler = get_lr_scheduler(args, optimizer)
    trainer = get_trainer(args, model, train_loader, optimizer,
                          lr_scheduler, device, tfboard)


    if args.debug:
        from lib.utils.debug_tools import get_rcnn_fg_bg_ratio
        trainer.add_event_handler(Events.STARTED,
                                  get_rcnn_fg_bg_ratio(args, model))

    trainer.run(train_loader, max_epochs=args.train.epochs)

    if is_main_process():
        tfboard.close()


if __name__ == '__main__':
    arg_parser = args_faster_rcnn_mae()
    args = arg_parser.parse_args(namespace=Nestedspace())
    print(args)
    print('\nStart traing MAE !\n')

    fn = get_mae_model
    main(args, fn)
