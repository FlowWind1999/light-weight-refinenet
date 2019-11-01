# general libs
import sys

sys.path.append("..")

import argparse
import logging
import time

# misc
import cv2

# pytorch libs
import torch.nn as nn

from PIL import Image
from torchvision import transforms

# custom libs
from src.config import *
from src.util import *
from src.datasets import ToTensor, Normalise
from utils.helpers import prepare_img


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


def getid(i):
    if i < 10:
        return '00000{}.png'.format(i)
    elif i < 100:
        return '0000{}.png'.format(i)
    elif i < 1000:
        return '000{}.png'.format(i)
    else:
        return '00{}.png'.format(i)


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
    print(RESULT_DIR+name)
    image.save(RESULT_DIR+name)

    img = cv2.imread(RESULT_DIR+name)
    img= cv2.Canny(img, 80, 150)
    cv2.imwrite(EDGE_DIR+name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def single_image(filepath):
    composed_val = transforms.Compose([Normalise(*NORMALISE_PARAMS),
                                       ToTensor()])
    image = np.array(Image.open(filepath))
    if len(image.shape) == 2:  # grayscale
        image = np.tile(image, [3, 1, 1]).transpose(1, 2, 0)
    sample = {'image': image, 'mask': image}
    input=composed_val(sample)['image']
    input=input.unsqueeze(0)
    input_var = torch.autograd.Variable(input).float()
    # Compute output
    output = segmenter(input_var)
    # 可视化
    #see_image(output, input_var, input, 0)
    image = nn.functional.interpolate(output,
                                      size=input_var.size()[2:], mode='bilinear', align_corners=False)
    image = image[0].data.cpu().numpy().transpose(1, 2, 0)
    image = cv2.resize(image, input.size()[2:][::-1], interpolation=cv2.INTER_CUBIC)
    cmap = np.load('../utils/cmap.npy')
    image = cmap[image.argmax(axis=2).astype(np.uint8)]
    image = transforms.ToPILImage()(image)

    name = os.path.basename(filepath)
    print(RESULT_DIR + name)
    image.save(RESULT_DIR + name)

    img = cv2.imread(RESULT_DIR + name)
    img = cv2.Canny(img, 80, 150)
    cv2.imwrite(EDGE_DIR + name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def images_list(path):
    composed_val = transforms.Compose([Normalise(*NORMALISE_PARAMS),
                                       ToTensor()])
    filelist = os.listdir(path)
    path = os.path.abspath(path)
    for i,item in enumerate(filelist):
        if item.endswith('.png'):
            filepath = os.path.join(path, item)
            single_image(filepath)
            '''
            image = np.array(Image.open(filepath))
            if len(image.shape) == 2:  # grayscale
                image = np.tile(image, [3, 1, 1]).transpose(1, 2, 0)
            sample = {'image': image, 'mask': image}
            input = composed_val(sample)['image']
            input = input.unsqueeze(0)
            input_var = torch.autograd.Variable(input).float()
            # Compute output
            output = segmenter(input_var)
            # 可视化
            see_image(output, input_var, input, i)
            '''

def single_24(filepath):
    cmap = np.load('../utils/cmap.npy')
    image = np.array(Image.open(filepath))
    orig_size = image.shape[:2][::-1]
    image = torch.tensor(prepare_img(image).transpose(2, 0, 1)[None]).float()
    # 分割
    segm = segmenter(image)[0].data.cpu().numpy().transpose(1, 2, 0)
    segm = cv2.resize(segm, orig_size, interpolation=cv2.INTER_CUBIC)
    segm = cmap[segm.argmax(axis=2).astype(np.uint8)]
    segm = transforms.ToPILImage()(segm)

    name = os.path.basename(filepath)
    print(RESULT_DIR + name)
    segm.save(RESULT_DIR + name)
    # 得到边缘
    img = cv2.imread(RESULT_DIR + name)
    img = cv2.Canny(img, 80, 150)
    cv2.imwrite(EDGE_DIR + name, img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

def test_24(path):
    filelist = os.listdir(path)
    path = os.path.abspath(path)
    #cmap = np.load('../utils/cmap.npy')
    for i, item in enumerate(filelist):
        if item.endswith('.png'):
            filepath = os.path.join(path, item)
            single_24(filepath)


def main():
    global args, logger, segmenter
    args = get_arguments()
    logger = logging.getLogger(__name__)
    ## Generate Segmenter ##
    segmenter = nn.DataParallel(
        create_segmenter(args.enc, args.enc_pretrained, args.num_classes[0])
    )
    segmenter.load_state_dict(torch.load('ckpt/checkpoint.pth.tar', 'cpu')['segmenter'])
    segmenter.eval()
    #single_image('../datasets/nyud/爆片绝缘子.png')
    images_list(r'./test/img')
    #test_24(r'./test/img')
    print('over')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    start = time.time()
    main()
    end = time.time()
    print("total used time is {} s".format(end - start))
