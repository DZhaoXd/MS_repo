import argparse
import numpy as np
import time
import sys
import os

# import torch
# import torch.nn as nn
# from torch.utils import data, model_zoo
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor, Parameter, context
import mindspore.common.initializer as init

from model.deeplab_multi import DeeplabMulti
from model.deeplabv2 import get_deeplab_v2
from model.discriminator import FCDiscriminator
from utils import CrossEntropy2d, SoftmaxCrossEntropyLoss, Softmax
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from evaluate_cityscapes import evaluation
from mindspore import amp

# context.set_context(mode=context.PYNATIVE_MODE)
# context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target='CPU')

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'DeepLab'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
# DATA_DIRECTORY = '/data/seaelm/datasets/GTA5'
DATA_DIRECTORY = r"E:\data\GTA5"
DATA_LIST_PATH = './dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '1280,720'
# INPUT_SIZE = '256,256'
# DATA_DIRECTORY_TARGET = '/data/zd/data/cityscape/cityscapes/Cityscapes'
DATA_DIRECTORY_TARGET = r"E:\datasets\Cityscapes\leftImg8bit_trainvaltest"
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
DEVKIT_DIR = r'./dataset/cityscapes_list'
INPUT_SIZE_TARGET = '1024,512'
# INPUT_SIZE_TARGET = '256,256'

LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234

# RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/DeepLab_resnet_pretrained_init-f81d91e8.pth'
RESTORE_FROM = r'D:\Files\GitHub\AdvSeg-Mindspore\model\Pretrain_DeeplabMulti.ckpt'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 5000
SNAPSHOT_DIR = './checkpoint/'
SAVE_RESULT_DIR = './result/cityscapes'

WEIGHT_DECAY = 0.0005
LEARNING_RATE_D = 1e-4
LAMBDA_SEG = 0.1
LAMBDA_ADV_TARGET1 = 0.0002
LAMBDA_ADV_TARGET2 = 0.001
GAN = 'Vanilla'

TARGET = 'cityscapes'
SET = 'train'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument('--devkit_dir', type=str, default=DEVKIT_DIR,
                        help='base directory of cityscapes.')
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--lambda-adv-target1", type=float, default=LAMBDA_ADV_TARGET1,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--lambda-adv-target2", type=float, default=LAMBDA_ADV_TARGET2,
                        help="lambda_adv for adversarial training.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument('--save_result_path', type=str, default=SAVE_RESULT_DIR,
                        help='保存中间分割结果的路径。')
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--device", type=str, default='cpu', choices=['cpu', 'gpu', 'ascend'],
                        help="choose device. ")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--gan", type=str, default=GAN,
                        help="choose the GAN objective.")
    parser.add_argument('--debug', action='store_true', default=False,
                        help='whether use debug mode.')
    parser.add_argument('--continue_train', type=str, default=None,
                        help='whether use continue train.')
    parser.add_argument('--not_val', action='store_false', default=True,
                        help='whether processing validation during the  training.')
    return parser.parse_args()


def split_checkpoint(checkpoint, split_list=None):
    if split_list == None:
        return checkpoint
    checkpoint_dict = {name: {} for name in split_list}
    for key, value in checkpoint.items():
        prefix = key.split('.')[0]
        if prefix not in checkpoint_dict:
            checkpoint_dict[key] = value.asnumpy()
            continue
        name = key.replace(prefix + '.', '')
        checkpoint_dict[prefix][name] = value
    return checkpoint_dict


class WithLossCellG(nn.Cell):
    def __init__(self, lambda_, net_G, net_D1, net_D2, loss_fn1, loss_fn2, size_source, size_target, batch_size=1,
                 num_classes=19):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.lambda_ = lambda_
        self.net_G = net_G
        self.net_D1 = net_D1
        self.net_D2 = net_D2
        self.net_G.set_grad(True)
        self.net_D1.set_grad(False)
        self.net_D2.set_grad(False)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.loss_fn1 = loss_fn1
        self.loss_fn2 = loss_fn2
        self.softmax = Softmax(axis=1)
        self.zeros_like = ops.ZerosLike()

    def construct(self, image_source, label, image_target):
        # time_1 = time.time()
        pred1, pred2 = self.net_G(image_source)
        pred1 = self.interp_source(pred1)
        pred2 = self.interp_source(pred2)

        loss_seg1 = self.loss_fn1(pred1, label)
        loss_seg2 = self.loss_fn1(pred2, label)
        pred1_target, pred2_target = self.net_G(image_target)

        pred1_target = self.interp_target(pred1_target)
        pred2_target = self.interp_target(pred2_target)
        pred1_target = self.softmax(pred1_target)
        pred2_target = self.softmax(pred2_target)

        out_D1 = self.net_D1(pred1_target)
        out_D2 = self.net_D2(pred2_target)
        source_label1 = self.zeros_like(out_D1)
        source_label2 = self.zeros_like(out_D2)
        loss_adv1 = self.loss_fn2(out_D1, source_label1)
        loss_adv2 = self.loss_fn2(out_D2, source_label2)

        loss = loss_seg2 + self.lambda_[0] * loss_seg1 + self.lambda_[2] * loss_adv2 + self.lambda_[1] * loss_adv1

        loss_seg1 = ops.stop_gradient(loss_seg1)
        loss_seg2 = ops.stop_gradient(loss_seg2)
        loss_adv1 = ops.stop_gradient(loss_adv1)
        loss_adv2 = ops.stop_gradient(loss_adv2)

        return loss, (loss_seg1, loss_seg2, loss_adv1, loss_adv2)


class WithLossCellD1(nn.Cell):
    def __init__(self, net_G, net_D1, loss_fn, size_source, size_target):
        super(WithLossCellD1, self).__init__(auto_prefix=True)
        self.net_G = net_G
        self.net_D1 = net_D1
        self.net_G.set_grad(False)
        self.net_D1.set_grad(True)
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.loss_fn = loss_fn
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.softmax = Softmax(axis=1)

    def construct(self, image_source, label, image_target):
        pred1, _ = self.net_G(image_source)
        pred1 = self.interp_source(pred1)
        pred1 = self.softmax(pred1)
        pred1 = ops.stop_gradient(pred1)

        pred1_target, _ = self.net_G(image_target)
        pred1_target = self.interp_target(pred1_target)
        pred1_target = self.softmax(pred1_target)
        pred1_target = ops.stop_gradient(pred1_target)

        out_s, out_t = self.net_D1(pred1), self.net_D1(pred1_target)
        label_s, label_t = self.zeros_like(out_s), self.ones_like(out_t)

        loss1 = self.loss_fn(out_s, label_s)
        loss2 = self.loss_fn(out_t, label_t)

        return (loss1 + loss2) / 2.0


class WithLossCellD2(nn.Cell):
    def __init__(self, net_G, net_D2, loss_fn, size_source, size_target):
        super(WithLossCellD2, self).__init__(auto_prefix=True)
        self.net_G = net_G
        self.net_D2 = net_D2
        self.net_G.set_grad(False)
        self.net_D2.set_grad(True)
        self.size_source = size_source
        self.size_target = size_target
        self.interp_source = ops.ResizeBilinear(size=(size_source[1], size_source[0]))
        self.interp_target = ops.ResizeBilinear(size=(size_target[1], size_target[0]))
        self.loss_fn = loss_fn
        self.zeros_like = ops.ZerosLike()
        self.ones_like = ops.OnesLike()
        self.softmax = Softmax(axis=1)

    def construct(self, image_source, label, image_target):
        _, pred2 = self.net_G(image_source)
        pred2 = self.interp_source(pred2)
        pred2 = self.softmax(pred2)
        pred2 = ops.stop_gradient(pred2)

        _, pred2_target = self.net_G(image_target)
        pred2_target = self.interp_target(pred2_target)
        pred2_target = self.softmax(pred2_target)
        pred2_target = ops.stop_gradient(pred2_target)

        out_s, out_t = self.net_D2(pred2), self.net_D2(pred2_target)
        label_s, label_t = self.zeros_like(out_s), self.ones_like(out_t)

        loss1 = self.loss_fn(out_s, label_s)
        loss2 = self.loss_fn(out_t, label_t)

        return (loss1 + loss2) / 2.0


class TrainOneStepCellG(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCellG, self).__init__(auto_prefix=False)
        self.network = network
        # self.network.set_grad()
        self.optimizer = optimizer
        self.weight = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        out = self.network(*inputs)
        grads = self.grad(self.network, self.weight)(*inputs)
        out = ops.functional.depend(out, self.optimizer(grads))
        return out


class TrainOneStepCellD(nn.Cell):
    def __init__(self, network, optimizer):
        super(TrainOneStepCellD, self).__init__(auto_prefix=False)
        self.network = network  # 定义前向网络
        # self.network.set_grad()  # 构建反向网络图
        self.optimizer = optimizer  # 定义优化器
        self.weight = self.optimizer.parameters  # 获取更新的权重
        self.grad = ops.GradOperation(get_by_list=True)  # 定义梯度计算方法

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weight)(*inputs)
        loss = ops.functional.depend(loss, self.optimizer(grads))
        return loss


def main():
    """Create the model and start the training."""
    args = get_arguments()

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    if args.device == 'ascend':
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
        # context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="Ascend")
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend")

    elif args.device == 'gpu':
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU")
    else:
        # context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="CPU")
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    print('设备：', args.device)

    if args.debug:
        args.batch_size = 1
        input_size = (128, 128)
        input_size_target = (128, 128)
        args.save_pred_every = 10
        args.num_steps_stop = 100
        args.not_val = False
        #

    # print(args)
    # [Part 1: Models]

    # Create network and load pretrain resnet parameters.

    # model = DeeplabMulti(num_classes=args.num_classes)
    model = get_deeplab_v2(num_classes=args.num_classes)
    print('model path:', args.restore_from)
    if args.restore_from:
        saved_state_dict = mindspore.load_checkpoint(args.restore_from)
        split_list = ['net_G', 'net_D1', 'net_D2']
        train_state_dict = split_checkpoint(saved_state_dict, split_list=split_list)
        mindspore.load_param_into_net(model, train_state_dict['net_G'])
        print('success load model !')

    # init D
    model_D1 = FCDiscriminator(num_classes=args.num_classes)
    model_D2 = FCDiscriminator(num_classes=args.num_classes)

    parameters = model.trainable_params()
    parameters_D1 = model_D1.trainable_params()
    parameters_D2 = model_D2.trainable_params()
    print('model_G:', len(parameters))
    print('model_D1:', len(parameters_D1))
    print('model_D2:', len(parameters_D2))

    # [Part 2: Optimizer and Loss function]
    learning_rate = nn.PolynomialDecayLR(learning_rate=args.learning_rate, end_learning_rate=1e-9,
                                         decay_steps=args.num_steps, power=args.power)
    optimizer = nn.SGD(model.trainable_params(),
                       learning_rate=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    learning_rate_D1 = nn.PolynomialDecayLR(learning_rate=args.learning_rate, end_learning_rate=1e-9,
                                            decay_steps=args.num_steps, power=args.power)
    optimizer_D1 = nn.Adam(model_D1.trainable_params(), learning_rate=learning_rate_D1, beta1=0.9, beta2=0.99)

    learning_rate_D2 = nn.PolynomialDecayLR(learning_rate=args.learning_rate, end_learning_rate=1e-9,
                                            decay_steps=args.num_steps, power=args.power)
    optimizer_D2 = nn.Adam(model_D2.trainable_params(), learning_rate=learning_rate_D2, beta1=0.9, beta2=0.99)

    loss_calc = SoftmaxCrossEntropyLoss()
    # loss_calc = CrossEntropy2d()
    bce_loss = nn.BCEWithLogitsLoss()

    #  [Whether continue train]
    iter_start = 0
    best_iou = 0.0
    if args.continue_train:
        filepath, filename = os.path.split(args.continue_train)
        target_path = filepath
        os.makedirs(target_path, exist_ok=True)
        logger = open(os.path.join(target_path, 'Train_log.log'), 'a')

        split_list = ['net_G', 'net_D1', 'net_D2']
        train_state_dict = mindspore.load_checkpoint(args.continue_train)
        train_state_dict = split_checkpoint(train_state_dict, split_list=split_list)
        iter_start = train_state_dict['iter']
        best_iou = train_state_dict['best_IoU']
        mindspore.load_param_into_net(model, train_state_dict['net_G'])
        mindspore.load_param_into_net(model_D1, train_state_dict['net_D1'])
        mindspore.load_param_into_net(model_D2, train_state_dict['net_D2'])
        optimizer.global_step.set_data(mindspore.Tensor([iter_start], dtype=mindspore.int32))
        optimizer_D1.global_step.set_data(mindspore.Tensor([iter_start], dtype=mindspore.int32))
        optimizer_D2.global_step.set_data(mindspore.Tensor([iter_start], dtype=mindspore.int32))
    else:
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        target_path = os.path.join(args.snapshot_dir, time_now)
        os.makedirs(target_path, exist_ok=True)
        logger = open(os.path.join(target_path, 'Train_log.log'), 'w')

    # [Part 3: dataset]

    # [Start] Create train dataset and target dataset iterator
    gta_genarator = GTA5DataSet(args.data_dir, args.data_list,
                                max_iters=args.num_steps * args.iter_size * args.batch_size,
                                crop_size=input_size, scale=args.random_scale,
                                mirror=args.random_mirror, mean=IMG_MEAN)

    gta_dataset = ds.GeneratorDataset(gta_genarator, shuffle=True, column_names=['image', 'label', 'size'])
    gta_dataset = gta_dataset.batch(batch_size=args.batch_size)
    train_iterator = gta_dataset.create_dict_iterator()

    cityscapes_generator = cityscapesDataSet(args.data_dir_target, os.path.join(args.devkit_dir, f'{args.set}.txt'),
                                             max_iters=args.num_steps * args.iter_size * args.batch_size,
                                             crop_size=input_size_target, scale=False,
                                             mirror=args.random_mirror, mean=IMG_MEAN,
                                             set=args.set)
    cityscapes_dataset = ds.GeneratorDataset(cityscapes_generator, shuffle=True,
                                             column_names=['image', 'size'])
    cityscapes_dataset = cityscapes_dataset.batch(batch_size=args.batch_size)
    target_iterator = cityscapes_dataset.create_dict_iterator()

    evaluation_generator = cityscapesDataSet(args.data_dir_target, os.path.join(args.devkit_dir, 'val.txt'),
                                             crop_size=input_size_target, scale=False,
                                             mirror=False, mean=IMG_MEAN,
                                             set='val')
    evaluation_dataset = ds.GeneratorDataset(evaluation_generator, shuffle=False,
                                             column_names=['image', 'size'])
    evaluation_dataset = evaluation_dataset.batch(batch_size=1)
    evaluation_iterator = evaluation_dataset.create_dict_iterator()
    # [Over] Create dataset iterator

    #######

    #######
    lambda_ = [args.lambda_seg, args.lambda_adv_target1, args.lambda_adv_target2]
    model_G_with_loss = WithLossCellG(lambda_=lambda_,
                                      net_G=model,
                                      net_D1=model_D1,
                                      net_D2=model_D2,
                                      loss_fn1=loss_calc,
                                      loss_fn2=bce_loss,
                                      size_source=input_size,
                                      size_target=input_size_target,
                                      batch_size=args.batch_size,
                                      num_classes=args.num_classes)
    model_D1_with_loss = WithLossCellD1(net_G=model,
                                        net_D1=model_D1,
                                        loss_fn=bce_loss,
                                        size_source=input_size,
                                        size_target=input_size_target)
    model_D2_with_loss = WithLossCellD2(net_G=model,
                                        net_D2=model_D2,
                                        loss_fn=bce_loss,
                                        size_source=input_size,
                                        size_target=input_size_target)

    # model_G_train = TrainOneStepCellG(model, optimizer)
    model_G_train = TrainOneStepCellG(model_G_with_loss, optimizer)
    model_D1_train = TrainOneStepCellD(model_D1_with_loss, optimizer_D1)
    model_D2_train = TrainOneStepCellD(model_D2_with_loss, optimizer_D2)

    model_G_train.set_train()
    model_D1_train.set_train()
    model_D2_train.set_train()

    # start train
    time_start_all = time.time()
    time_start_one = time.time()
    time_start_log = time.time()

    print(f'训练启动：',
          '\n开始的代数：', iter_start,
          '\n最高IoU：', best_iou,
          '保存地址：', target_path
          )

    for i_iter in range(iter_start, args.num_steps):

        s_data = next(train_iterator)
        image_s, label_s = s_data['image'], s_data['label']
        t_data = next(target_iterator)
        image_t = t_data['image']

        image_s, label_s = Tensor(image_s), Tensor(label_s)
        image_t = Tensor(image_t)

        loss, (loss_seg1, loss_seg2, loss_adv1, loss_adv2) = model_G_train(image_s, label_s, image_t)
        (loss_seg1, loss_seg2, loss_adv1, loss_adv2) = \
            map(lambda x: x.asnumpy(), (loss_seg1, loss_seg2, loss_adv1, loss_adv2))

        loss_D1 = model_D1_train(image_s, label_s, image_t).asnumpy()
        loss_D2 = model_D2_train(image_s, label_s, image_t).asnumpy()

        time_end_one = time.time()
        print('iter = {0:8d}/{1:8d}, loss_seg1 = {2:.6f} loss_seg2 = {3:.6f} loss_adv1 = {4:.6f}, loss_adv2 = {5:.6f} '
              'loss_D1 = {6:.6f} loss_D2 = {7:.6f} time:{8:.6f}'.format(
            i_iter, args.num_steps, loss_seg1, loss_seg2, loss_adv1, loss_adv2, loss_D1, loss_D2,
            time_end_one - time_start_one))
        time_start_one = time_end_one
        if i_iter % 1 == 0:
            time_end_log = time.time()
            logger.write(
                'iter = {0:8d}/{1:8d}, loss_seg1 = {2:.6f} loss_seg2 = {3:.6f} loss_adv1 = {4:.6f}, loss_adv2 = {5:.6f} '
                'loss_D1 = {6:.6f} loss_D2 = {7:.6f} time:{8:.6f}\n'.format(
                    i_iter, args.num_steps, loss_seg1, loss_seg2, loss_adv1, loss_adv2, loss_D1, loss_D2,
                    time_end_log - time_start_log))
            time_start_log = time_end_log

        if (i_iter + 1) % args.save_pred_every == 0:
            print('val checkpoint ...')

            if args.not_val:
                miou = evaluation(model, evaluation_iterator, ops.ResizeBilinear(size=(1024, 2048)),
                                  args.data_dir_target,
                                  args.save_result_path, args.devkit_dir, logger=logger, save=False)
                miou = float(miou)
            else:
                miou = -0.1

            checkpoint_path = os.path.join(target_path, 'GTA5_' + str(i_iter + 1) + '.ckpt')

            # param_list用于整合多个模型的参数，不同模型的参数通过name_prefix添加前缀来区别，并且加载后用于划分。
            # 根据需求此部分进行修改。
            param_list = [{'name': name, 'data': param} for name, param in
                          model.parameters_and_names(name_prefix='net_G')]
            for name, param in model_D1.parameters_and_names(name_prefix='net_D1'):
                param_list.append({'name': name, 'data': param})
            for name, param in model_D2.parameters_and_names(name_prefix='net_D2'):
                param_list.append({'name': name, 'data': param})

            # append_dict 用于保存中间的各项参数情况，官方介绍仅值类型支持int，float，bool。
            append_dict = {'iter': int(optimizer.global_step.asnumpy()[0]),
                           'mIoU': float(miou),
                           'best_IoU': float(best_iou) if miou < best_iou else float(miou)}
            mindspore.save_checkpoint(param_list, checkpoint_path, append_dict=append_dict)

            if miou > best_iou:
                best_iou = miou
                checkpoint_path = os.path.join(target_path, 'GTA5_best.ckpt')
                mindspore.save_checkpoint(param_list, checkpoint_path, append_dict=append_dict)
            print('Best mIoU:', best_iou)
            logger.write(f'Best mIoU:{best_iou}\n')
            if i_iter >= args.num_steps_stop - 1:
                checkpoint_path = os.path.join(target_path, 'GTA5_Over.ckpt')
                mindspore.save_checkpoint(param_list, checkpoint_path, append_dict=append_dict)
                break
    print('Train Over ! Save Over model ')

    time_end_all = time.time()
    logger.write(f'训练总用时：{time_end_all - time_start_all}')
    logger.close()


if __name__ == '__main__':
    main()
    # a = r"E:\datasets\Cityscapes\leftImg8bit_trainvaltest\GTA5_D2_3.ckpt"
    # file_path, filename = os.path.split(a)
    # # print(file_path)
    # print(filename.split('.')[0])
    # filename=filename.split('.')[0]
    # iter = int(filename.split('_')[-1])
    # print(iter)
