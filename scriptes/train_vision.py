"""Train YOLOv3 with random shapes."""
import argparse
import logging
import os
import time
import warnings

import gluoncv as gcv
import mxnet as mx
from gluoncv import utils as gutils
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.dataloader import RandomTransformDataLoader
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultTrainTransform, YOLO3DefaultValTransform
from gluoncv.model_zoo import get_model
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxnet import autograd
from mxnet import gluon
from mxnet.contrib import amp

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

import rmai

gcv.utils.check_version('0.7.0')


def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')

    parser.add_argument('--network', type=str, default='mobilenet1.0',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=320,
                        help="Input data shape for evaluation, use 320, 416, 608... " +
                             "Training is with random shapes from (320 to 608).")
    parser.add_argument('--dtype', type=str, default='float16',
                        help='')
    parser.add_argument('--pretrain-dataset', type=str, default='voc',
                        help='Training dataset. voc or coco.')

    parser.add_argument('--dataset', type=str, default='dji',
                        help='Training dataset. Now support dji.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=8, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--pin_memory', type=bool, default=True,
                        help='Number of data workers, you can use larger '
                             'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./yolo3_xxx_0123.params')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Training epochs.')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training mini-batch size')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--max-update', type=int, default=10,
                        help='maximum number of updates before the decay reaches 0.')
    parser.add_argument('--base-lr', type=float, default=0.001,
                        help='base learning rate. default is 0.001')
    parser.add_argument('--final-lr', type=float, default=0.00001,
                        help='final learning rate after all steps. default is 0.0001')
    parser.add_argument('--warmup-epochs', type=int, default=2,
                        help='number of warmup steps used before this scheduler starts decay. default is 2.')
    parser.add_argument('--warmup-begin-lr', type=float, default=0.0001,
                        help='if using warmup, the learning rate from which it starts warming up.')
    parser.add_argument('--step-epochs', type=str, default='3,4',
                        help='The list of steps to schedule a change.')
    parser.add_argument('--factor', type=float, default=0.1,
                        help='The factor to change the learning rate.')
    parser.add_argument('--pwr', type=int, default=2,
                        help='power of the decay term as a function of the current number of updates.')

    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')

    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')

    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--syncbn', action='store_true',
                        help='Use synchronize BN across devices.')
    parser.add_argument('--no-random-shape', action='store_true',
                        help='Use fixed size(data-shape) throughout the training, which will be faster '
                             'and require less memory. However, final model will be slightly worse.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether to enable mixup.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')
    parser.add_argument('--label-smooth', action='store_true', help='Use label smoothing.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')
    parser.add_argument('--horovod', action='store_true',
                        help='Use MXNet Horovod for distributed training. Must be run with OpenMPI. '
                             '--gpus is ignored when using --horovod.')

    return parser.parse_args()


def get_dataset(args):
    # Get dataset.
    if args.dataset.lower() == 'dji':
        train_set = rmai.DJIROCODetection(
            splits=[('central', 'trainval')]
        )

        val_set = rmai.DJIROCODetection(
            splits=[('south', 'test')])  # ('central', 'test'), ('final', 'test'), ('north', 'test')

        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_set.classes)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(args.dataset))

    if args.mixup:
        from gluoncv.data import MixupDetection
        train_set = MixupDetection(train_set)

    return train_set, val_set, val_metric


def get_dataloader(net, train_set, val_set, batch_size, args):
    # Get dataloader.
    width, height = args.data_shape, args.data_shape

    # stack image, all targets generated
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))

    if args.no_random_shape:
        train_loader = gluon.data.DataLoader(
            train_set.transform(YOLO3DefaultTrainTransform(width, height, net, mixup=args.mixup)),
            batch_size=batch_size, shuffle=True, batchify_fn=batchify_fn,
            last_batch='rollover', num_workers=args.num_workers, pin_memory=False)
    else:
        transform_fns = [YOLO3DefaultTrainTransform(x * 32, x * 32, net, mixup=args.mixup) for x in range(10, 20)]
        train_loader = RandomTransformDataLoader(
            transform_fns, train_set, interval=10,
            batch_size=batch_size, shuffle=True, batchify_fn=batchify_fn,
            last_batch='rollover', num_workers=args.num_workers, pin_memory=False)

    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_set.transform(YOLO3DefaultValTransform(width, height)),
        batch_size=batch_size, shuffle=True, batchify_fn=val_batchify_fn,
        last_batch='keep', num_workers=args.num_workers, pin_memory=False)

    return train_loader, val_loader


def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def validate(net, val_data, ctx, eval_metric):
    """Test on validation dataset."""
    eval_metric.reset()
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()


def train(net, train_dloader, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if args.label_smooth:
        net._target_generator._label_smooth = True

    # learn rate scheduler.
    num_batches = len(train_dloader)
    step = [int(i) * num_batches for i in args.step_epochs.split(',')]
    warmup_steps = args.warmup_epochs * num_batches

    if args.lr_mode == 'step':
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(
            step, args.factor, args.base_lr, warmup_steps, args.warmup_begin_lr)

    elif args.lr_mode == 'poly':
        lr_scheduler = mx.lr_scheduler.PolyScheduler(
            args.max_update, args.base_lr, args.pwr, args.final_lr, warmup_steps, args.warmup_begin_lr)

    elif args.lr_mode == 'cosine':
        lr_scheduler = mx.lr_scheduler.CosineScheduler(
            args.max_update, args.base_lr, args.final_lr, warmup_steps, args.warmup_begin_lr)
    else:
        raise NotImplementedError('lr-mode: {} not implemented.'.format(args.lr_mode))

    if args.horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
            net.collect_params(), 'sgd',
            {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler})
    else:
        trainer = gluon.Trainer(
            net.collect_params(), 'sgd',
            {'multi_precision': True, 'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler},
            kvstore='local', update_on_kvstore=(False if args.amp else None))

    if args.amp:
        amp.init_trainer(trainer)

    # metrics
    obj_metrics = mx.metric.Loss('ObjLoss')
    center_metrics = mx.metric.Loss('BoxCenterLoss')
    scale_metrics = mx.metric.Loss('BoxScaleLoss')
    cls_metrics = mx.metric.Loss('ClassLoss')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    for epoch in range(args.start_epoch, args.epochs):
        tic = time.time()
        btic = time.time()
        mx.nd.waitall()
        net.hybridize()
        for i, batch in enumerate(train_dloader):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            # objectness, center_targets, scale_targets, weights, class_targets
            fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=ctx, batch_axis=0) for it in range(1, 6)]
            gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=ctx, batch_axis=0)
            sum_losses = []
            obj_losses = []
            center_losses = []
            scale_losses = []
            cls_losses = []
            with autograd.record():
                for ix, x in enumerate(data):
                    obj_loss, center_loss, scale_loss, cls_loss = net(x, gt_boxes[ix],
                                                                      *[ft[ix] for ft in fixed_targets])
                    sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                    obj_losses.append(obj_loss)
                    center_losses.append(center_loss)
                    scale_losses.append(scale_loss)
                    cls_losses.append(cls_loss)
                if args.amp:
                    with amp.scale_loss(sum_losses, trainer) as scaled_loss:
                        autograd.backward(scaled_loss)
                else:
                    autograd.backward(sum_losses)
            trainer.step(batch_size)
            if (not args.horovod or hvd.rank() == 0):
                obj_metrics.update(0, obj_losses)
                center_metrics.update(0, center_losses)
                scale_metrics.update(0, scale_losses)
                cls_metrics.update(0, cls_losses)
                if args.log_interval and not (i + 1) % args.log_interval:
                    name1, loss1 = obj_metrics.get()
                    name2, loss2 = center_metrics.get()
                    name3, loss3 = scale_metrics.get()
                    name4, loss4 = cls_metrics.get()
                    logger.info(
                        '[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                            epoch, i, trainer.learning_rate, args.batch_size / (time.time() - btic), name1, loss1,
                            name2, loss2, name3, loss3, name4, loss4))
                btic = time.time()

        if (not args.horovod or hvd.rank() == 0):
            name1, loss1 = obj_metrics.get()
            name2, loss2 = center_metrics.get()
            name3, loss3 = scale_metrics.get()
            name4, loss4 = cls_metrics.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time() - tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)


if __name__ == '__main__':
    args = parse_args()

    if args.amp:
        amp.init()

    if args.horovod:
        if hvd is None:
            raise SystemExit("Horovod not found, please check if you installed it correctly.")
        hvd.init()

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('yolo3', args.network, args.pretrain_dataset))
    args.save_prefix += net_name + '_' + args.dataset

    # use sync bn if specified
    if args.syncbn and len(ctx) > 1:
        # TODO: Try use more combination.
        network = get_model(net_name,
                            pretrained=True,
                            norm_layer=gluon.contrib.nn.SyncBatchNorm,
                            norm_kwargs={'num_devices': len(ctx)})
        async_net = get_model(net_name, pretrained=False)  # Used by CPU worker.
    else:
        network = get_model(net_name, pretrained=True)
        async_net = network
    if args.resume.strip():
        network.load_parameters(args.resume.strip())
        async_net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            network.initialize()
            async_net.initialize()

    train_dataset, val_dataset, val_metric = get_dataset(args)
    network.reset_class(train_dataset.classes)
    # network.cast(args.dtype)

    # training data
    batch_size = (args.batch_size // hvd.size()) if args.horovod else args.batch_size
    train_dataloader, val_dataloader = get_dataloader(async_net, train_dataset, val_dataset, batch_size, args)

    # training
    train(network, train_dataloader, val_dataloader, val_metric, ctx, args)

    # TODO: Export to onnx.
