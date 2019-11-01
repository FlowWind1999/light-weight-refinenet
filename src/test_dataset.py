import sys
sys.path.append("..")

import argparse
import logging
import time

# misc
import cv2
import numpy as np

# pytorch libs
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# custom libs
from src.config import *
#from src.miou_utils import compute_iu, fast_cm
from src.util import *

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full Pipeline Training")

    # Dataset
    parser.add_argument("--train-dir", type=str, default=TRAIN_DIR,
                        help="Path to the training set directory.")
    parser.add_argument("--val-dir", type=str, default=VAL_DIR,
                        help="Path to the validation set directory.")
    parser.add_argument("--train-list", type=str, nargs='+', default=TRAIN_LIST,
                        help="Path to the training set list.")
    parser.add_argument("--val-list", type=str, nargs='+', default=VAL_LIST,
                        help="Path to the validation set list.")
    parser.add_argument("--shorter-side", type=int, nargs='+', default=SHORTER_SIDE,
                        help="Shorter side transformation.")
    parser.add_argument("--crop-size", type=int, nargs='+', default=CROP_SIZE,
                        help="Crop size for training,")
    parser.add_argument("--normalise-params", type=list, default=NORMALISE_PARAMS,
                        help="Normalisation parameters [scale, mean, std],")
    parser.add_argument("--batch-size", type=int, nargs='+', default=BATCH_SIZE,
                        help="Batch size to train the segmenter model.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for pytorch's dataloader.")
    parser.add_argument("--num-classes", type=int, nargs='+', default=NUM_CLASSES,
                        help="Number of output classes for each task.")
    parser.add_argument("--low-scale", type=float, nargs='+', default=LOW_SCALE,
                        help="Lower bound for random scale")
    parser.add_argument("--high-scale", type=float, nargs='+', default=HIGH_SCALE,
                        help="Upper bound for random scale")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="Label to ignore during training")

    # Encoder
    parser.add_argument("--enc", type=str, default=ENC,
                        help="Encoder net type.")
    parser.add_argument("--enc-pretrained", type=bool, default=ENC_PRETRAINED,
                        help='Whether to init with imagenet weights.')
    # General
    parser.add_argument("--evaluate", type=bool, default=EVALUATE,
                        help='If true, only validate segmentation.')
    parser.add_argument("--freeze-bn", type=bool, nargs='+', default=FREEZE_BN,
                        help='Whether to keep batch norm statistics intact.')
    parser.add_argument("--num-segm-epochs", type=int, nargs='+', default=NUM_SEGM_EPOCHS,
                        help='Number of epochs to train for segmentation network.')
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY,
                        help='Print information every often.')
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help='Seed to provide (near-)reproducibility.')
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Path to directory for storing checkpoints.")
    parser.add_argument("--ckpt-path", type=str, default=CKPT_PATH,
                        help="Path to the checkpoint file.")
    parser.add_argument("--val-every", nargs='+', type=int, default=VAL_EVERY,
                        help="How often to validate current architecture.")

    # Optimisers
    parser.add_argument("--lr-enc", type=float, nargs='+', default=LR_ENC,
                        help="Learning rate for encoder.")
    parser.add_argument("--lr-dec", type=float, nargs='+', default=LR_DEC,
                        help="Learning rate for decoder.")
    parser.add_argument("--mom-enc", type=float, nargs='+', default=MOM_ENC,
                        help="Momentum for encoder.")
    parser.add_argument("--mom-dec", type=float, nargs='+', default=MOM_DEC,
                        help="Momentum for decoder.")
    parser.add_argument("--wd-enc", type=float, nargs='+', default=WD_ENC,
                        help="Weight decay for encoder.")
    parser.add_argument("--wd-dec", type=float, nargs='+', default=WD_DEC,
                        help="Weight decay for decoder.")
    parser.add_argument("--optim-dec", type=str, default=OPTIM_DEC,
                        help="Optimiser algorithm for decoder.")
    return parser.parse_args()

def create_segmenter(
    net, pretrained, num_classes
    ):
    """Create Encoder; for now only ResNet [50,101,152]"""
    from models.resnet import rf_lw50, rf_lw101, rf_lw152
    if str(net) == '50':
        return rf_lw50(num_classes, imagenet=pretrained)
    elif str(net) == '101':
        return rf_lw101(num_classes, imagenet=pretrained)
    elif str(net) == '152':
        return rf_lw152(num_classes, imagenet=pretrained)
    else:
        raise ValueError("{} is not supported".format(str(net)))

def create_loaders(
    train_dir, val_dir, train_list, val_list,
    shorter_side, crop_size, low_scale, high_scale,
    normalise_params, batch_size, num_workers, ignore_label
    ):
    """
    Args:
      train_dir (str) : path to the root directory of the training set.
      val_dir (str) : path to the root directory of the validation set.
      train_list (str) : path to the training list.
      val_list (str) : path to the validation list.
      shorter_side (int) : parameter of the shorter_side resize transformation.
      crop_size (int) : square crop to apply during the training.
      low_scale (float) : lowest scale ratio for augmentations.
      high_scale (float) : highest scale ratio for augmentations.
      normalise_params (list / tuple) : img_scale, img_mean, img_std.
      batch_size (int) : training batch size.
      num_workers (int) : number of workers to parallelise data loading operations.
      ignore_label (int) : label to pad segmentation masks with

    Returns:
      train_loader, val loader

    """
    # Torch libraries
    from torchvision import transforms
    from torch.utils.data import DataLoader, random_split
    # Custom libraries
    from src.datasets import NYUDataset as Dataset
    from src.datasets import Pad, RandomCrop, RandomMirror, ResizeShorterScale, ToTensor, Normalise

    ## Transformations during training ##
    composed_trn = transforms.Compose([ResizeShorterScale(shorter_side, low_scale, high_scale),
                                    Pad(crop_size, [123.675, 116.28 , 103.53], ignore_label),
                                    RandomMirror(),
                                    RandomCrop(crop_size),
                                    Normalise(*normalise_params),
                                    ToTensor()])
    composed_val = transforms.Compose([Normalise(*normalise_params),
                                    ToTensor()])
    ## Training and validation sets ##
    trainset = Dataset(data_file=train_list,
                       data_dir=train_dir,
                       transform_trn=composed_trn,
                       transform_val=composed_val)

    valset = Dataset(data_file=val_list,
                         data_dir=val_dir,
                         transform_trn=None,
                         transform_val=composed_val)
    logger.info(" Created train set = {} examples, val set = {} examples"
                .format(len(trainset), len(valset)))
    ## Training and validation loaders ##
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(valset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    return train_loader, val_loader

def getid(i):
    if i<10:
        return RESULT_DIR+'00000{}.png'.format(i)
    elif i<100:
        return RESULT_DIR+'0000{}.png'.format(i)
    elif i<1000:
        return RESULT_DIR+'000{}.png'.format(i)
    else:
        return RESULT_DIR+'00{}.png'.format(i)

def see_image(output, input_var, input, id):
    # 可视化
    image = nn.functional.interpolate(output,
            size=input_var.size()[2:], mode='bilinear', align_corners=False)
    image = image[0].data.cpu().numpy().transpose(1, 2, 0)
    image = cv2.resize(image, input.size()[2:][::-1], interpolation=cv2.INTER_CUBIC)
    cmap = np.load('../utils/cmap.npy')
    image = cmap[image.argmax(axis=2).astype(np.uint8)]
    image = transforms.ToPILImage()(image)
    name = getid(id)
    print(name)
    image.save(name)

def main():
    global args, logger
    args = get_arguments()
    logger = logging.getLogger(__name__)
    ## Add args ##
    args.num_stages = len(args.num_classes)
    ## Generate Segmenter ##
    segmenter = nn.DataParallel(
        create_segmenter(args.enc, args.enc_pretrained, args.num_classes[0])
    )
    segmenter.load_state_dict(torch.load('ckpt/checkpoint.pth.tar','cpu')['segmenter'])

    #for task_idx in range(args.num_stages):
    start = time.time()
    ## Create dataloaders ##
    train_loader, val_loader = create_loaders(args.train_dir,
                                                args.val_dir,
                                                args.train_list[0],
                                                args.val_list[0],
                                                args.shorter_side[0],
                                                args.crop_size[0],
                                                args.low_scale[0],
                                                args.high_scale[0],
                                                args.normalise_params,
                                                args.batch_size[0],
                                                args.num_workers,
                                                args.ignore_label)

    val_loader.dataset.set_stage('val')
    segmenter.eval()
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            input = sample['image']
            input_var = torch.autograd.Variable(input).float()
            # Compute output
            output = segmenter(input_var)
            see_image(output, input_var, input, i)
    print('over')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
