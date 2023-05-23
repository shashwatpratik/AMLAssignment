import os
import os.path as osp
import sys
import warnings
import time

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

import models.CNN
from Settings import Settings
from args import argument_parser, optimizer_kwargs
from dataset.MIDataset import MIDataset
from dataset import load_dataset
from dataset import transforms
from losses import CrossEntropyLoss
from optimizers import init_optimizer
from utils.avgmeter import AverageMeter

from utils.generaltools import set_random_seed

from models.CNN import net3D
from models.my_model import dummy_model
from models.M3T import M3T
from models.MedResnet import MedResnet

# global variables
from utils.iotools import check_isfile
from utils.loggers import Logger
from utils.torchtools import load_pretrained_weights, resume_from_checkpoint, save_checkpoint, accuracy,count_num_param

parser = argument_parser()
args = parser.parse_args()

def main():
    global args
    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = "log_test.txt" if args.evaluate else "log_train.txt"
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    print(f"==========\nArgs:{args}\n==========")
    if use_gpu:
        print(f"Currently using GPU {args.gpu_devices}")
        cudnn.benchmark = True
    else:
        warnings.warn("Currently using CPU, however, GPU is highly recommended")


    train_ds,_ = load_dataset.build_dataset(Settings.SPLIT_PATH,type='train',transforms=transforms.monai_transforms)
    val_ds,_ = load_dataset.build_dataset(Settings.SPLIT_PATH,type='val',transforms=transforms.monai_transforms)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size= args.train_batch_size)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size= args.test_batch_size)

    
    if args.arch == 'M3T':
        model = M3T(num_classes=3, in_channel=1, out_channel_3D=32, out_dim_2D=256, use_gpu=use_gpu)
    else:
        model = MedResnet(num_classes=3, in_channel=1, out_channel_3D=32, out_dim_2D=256, use_gpu=use_gpu)

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)

    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion_CELoss = nn.CrossEntropyLoss()
    criterion_xent = nn.DataParallel(criterion_CELoss).cuda() if use_gpu else criterion_CELoss

    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(
            args.resume, model, optimizer=optimizer
        )
    if args.evaluate:
        test(model, val_loader, criterion_xent, use_gpu)
        return

    for epoch in range(args.start_epoch, args.max_epoch):
        train(
            epoch,
            model,
            criterion_xent,
            optimizer,
            train_loader,
            use_gpu,
        )
        if (
                (epoch + 1) > args.start_eval
                and args.eval_freq > 0
                and (epoch + 1) % args.eval_freq == 0
                or (epoch + 1) == args.max_epoch
        ):
            test(model,val_loader,criterion_xent,use_gpu)
            save_checkpoint(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "optimizer": optimizer.state_dict(),
                },
                args.save_dir,
            )
    print('------------Process End---------------')


def train(epoch, model, criterion, optimizer, train_loader, use_gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()

    #torch.autograd.set_detect_anomaly(True)

    model.train()
    for p in model.parameters():
        p.requires_grad = True

    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if use_gpu:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        losses.update(loss.item(), 3)
        accs.update(accuracy(outputs, targets).item())

        log_results('train',data_time.val,epoch+1,batch_time.val,batch_idx+1,losses.val,losses.avg,accs.val,accs.avg)
    return True

def test(model, val_loader, criterion, use_gpu):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    model.eval()
    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            data_time.update(time.time() - end)

            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()


            outputs = model(inputs)


            loss = criterion(outputs, targets)
            batch_time.update(time.time() - end)
            losses.update(loss.item(), 3)
            accs.update(accuracy(outputs, targets).item())
            log_results('val', data_time.val, 1, batch_time.val, batch_idx + 1, losses.val,losses.avg,accs.val,accs.avg)
    return True

def log_results(type, time, epoch, batch_time, batch, loss, loss_avg, acc, acc_avg):
    print(f'{type},{time:.3f},{epoch},{batch_time:.3f},{batch},{loss:.3f},{loss_avg:.3f},{acc:.2f},{acc_avg:.2f}')

if __name__ == "__main__":
    main()