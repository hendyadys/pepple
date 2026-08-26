import os, sys, argparse, cv2, shutil, time, random, glob, json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models
# from torchsummary import summary
from torch.optim import Adam, lr_scheduler

# architectures
from myResnet import resnet101, resnet50, resnet34, resnet18
# data manipulation
from PeppleDataset import PeppleDataset, get_imgs_for_split
from PeppleDataset_cutMix import PeppleDataset_cutMix
from PeppleDataset_mixup import PeppleDataset_mixup

AC_CLASSES = 6
CLASSIFICATION_FOLDER = os.path.join('classification')
best_acc1 = 0
IMG_SIZE = 512
# IMG_SIZE = 1024
IMG_ROW = 512


# regular convolutions instead of sparse convolutions
def get_model(model_name, num_channels=3, num_classes=AC_CLASSES, input_rows=512, input_cols=512):
    if (input_rows < 225):  # original images 224
        adj_factor = 1
    elif input_rows==320:
        adj_factor = 16
    elif input_rows==512 and input_cols==1024:
        adj_factor = 260
    elif input_rows==512:
        adj_factor = 100
    elif input_rows==1024:
        adj_factor = 676

    print('adj_factor=', adj_factor)

    # pretrained on imagenet - only works for num_channels=3
    if model_name=='resnet101':
        model = resnet101(pretrained=False, num_channels=num_channels, num_classes=num_classes, adj_factor=adj_factor)
    elif model_name=='resnet50':
        model = resnet50(pretrained=False, num_channels=num_channels, num_classes=num_classes, adj_factor=adj_factor)
    elif model_name == 'resnet34':
        model = resnet34(pretrained=False, num_channels=num_channels, num_classes=num_classes, adj_factor=adj_factor)
    elif model_name=='resnet18':
        model = resnet18(pretrained=False, num_channels=num_channels, num_classes=num_classes, adj_factor=adj_factor)

    if torch.cuda.is_available():
        device = torch.device('cuda')   # this defaults to torch.cuda.current_device (default device=)
    else:
        device = torch.device('cpu')
    # FIXME - torchsummary.summary() is buggy for device!=cuda0 implied.
    # summary(model.to(device), (num_channels, input_rows, input_cols), device="cuda:{}".format(args.gpu))
    return model


# https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501/2
# define "soft" cross-entropy with pytorch tensor operations
def softXEnt(input, target):
    logprobs = torch.nn.functional.log_softmax (input, dim = 1)
    temp = -(target * logprobs).sum() / input.shape[0]
    # print('l.shape=', logprobs.shape, logprobs.dtype, 't*l=', (target * logprobs).shape, temp.shape, temp.dtype, temp)
    return temp


def train_wrapper():
    global best_acc1

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if torch.cuda.is_available():
        device = torch.device('cuda')   # this defaults to torch.cuda.current_device (default device=)
    else:
        device = torch.device('cpu')
    net = get_model(args.arch_name, num_channels=3, num_classes=AC_CLASSES, input_rows=IMG_ROW, input_cols=IMG_SIZE)
    net = torch.nn.DataParallel(net)

    if torch.cuda.is_available():
        net = net.cuda()

    if args.cutMix or args.mixup:
        criterion = softXEnt
        criterion_valid = torch.nn.CrossEntropyLoss().cuda()
    else:
        if args.cls_weighted:
            class_weights = torch.FloatTensor([0.2, 0.9, 0.9, 0.9, 0.9, 0.9]).cuda()
            # criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)
            criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
        else:
            # criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
            criterion = torch.nn.CrossEntropyLoss().cuda()
        criterion_valid = criterion

    optimizer = Adam(net.parameters(), args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(args.resume)
            checkpoint = load_weights(args.resume, args.gpu)  # checks cuda.is_available()
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            # print('state_dict', checkpoint['state_dict']['device'])
            net.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    train_img_labels, valid_img_labels, test_img_labels = get_imgs_for_split(args.region, nrow=IMG_ROW, ncol=IMG_SIZE)
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    # training image transformations
    transform = transforms.Compose([
        # transforms.ColorJitter(brightness=.2, contrast=.2, saturation=0.2, hue=.1),
        transforms.RandomAffine(degrees=(0, 360), scale=(0.9, 1.1), shear=(.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    val_dataset = PeppleDataset(valid_img_labels, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    save_folder = os.path.join(CLASSIFICATION_FOLDER, 'torch_runs')
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    hist_file = 'history_{}_lr{}_w{}.txt'.format(args.region, args.lr, args.cls_weighted)
    valid_file = 'valid_{}_lr{}_w{}.txt'.format(args.region, args.lr, args.cls_weighted)

    with open(os.path.join(save_folder, hist_file), "a") as fout:   # to allow args.resume to write properly
        with open(os.path.join(save_folder, valid_file), "a") as fvalid:    # to allow args.resume to write properly
            # fout.write("epoch\tstep\tloss\tavgloss\tlr\n")
            fvalid.write("epoch\tacc1\tacc2\tval_loss\t\n")

            for epoch in range(args.start_epoch, args.epochs):
                if args.cutMix:
                    print('using cutMix augmentation')
                    train_dataset = PeppleDataset_cutMix(train_img_labels, transform=transform)
                elif args.mixup:
                    print('using mixup augmentation')
                    train_dataset = PeppleDataset_mixup(train_img_labels, transform=transform)
                else:
                    print('standard data loading')
                    train_dataset = PeppleDataset(train_img_labels, transform=transform)

                # shuffle to prevent iterations being unbalanced
                train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

                # train for one epoch
                train(train_loader, net, criterion, optimizer, epoch, hist_out=fout, args=args)

                # evaluate on validation set
                acc1, acc2, val_loss, _ = validate(val_loader, net, criterion_valid)     # already averaged
                fvalid.write("%d\t%0.4f\t%0.4f\t%0.4f\t\n" % (epoch + 1, acc1, acc2, val_loss))
                print("Epoch [%d] final val_acc1=%0.4f; val_acc2=%0.4f;  val_loss = %0.4f" % (epoch + 1, acc1, acc2, val_loss))

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1
                print('validation accuracy', 'acc=', acc1, 'prev_best=', best_acc1)
                # is_best = False
                best_acc1 = max(acc1, best_acc1)

                if is_best:
                    save_name = '{}_{}_lr{}_w{}_{}'.format(args.arch_name, args.region, args.lr, args.cls_weighted, epoch)
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'lr':args.lr,
                        'state_dict': net.state_dict(), # embedded state_dict!
                        'best_acc1': best_acc1,
                        'optimizer': optimizer.state_dict(),
                    }, is_best, folder=save_folder, filename=save_name)
    return


def train(train_loader, model, criterion, optimizer, epoch, hist_out=None, args=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (imgs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # combine input with mask
        if torch.cuda.is_available():
            # input = imgs.cuda(args.gpu, non_blocking=True)
            input = imgs.cuda()
        else:
            input = imgs

        if torch.cuda.is_available():
            targets = targets.cuda()

        # # compute output
        output = model(input)
        # print('train', output.shape, targets.shape, output.type, targets.type, criterion)
        loss = criterion(output, targets)
        # print('train', output.shape, targets.shape, output.type, targets.type, criterion, loss.shape, loss)
        # sys.exit()

        # # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        if not (args.cutMix or args.mixup):
            acc1, acc2 = accuracy(output, targets, topk=(1, 2))
            top1.update(acc1[0], input.size(0))
            top2.update(acc2[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.cutMix or args.mixup:
            log_msg = 'Epoch: [{0}][{1}/{2}]\t ' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses)
        else:
            log_msg = 'Epoch: [{0}][{1}/{2}]\t ' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                      'Acc@2 {top2.val:.3f} ({top2.avg:.3f})\n'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top2=top2)

        if i % args.log_interval == 0:
            print(log_msg)
        if hist_out is not None:
            hist_out.write(log_msg)
    return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    output_all = []
    with torch.no_grad():
        end = time.time()
        for i, (imgs, targets) in enumerate(val_loader):
            # combine input with mask
            input = imgs.cuda()
            if torch.cuda.is_available():
                targets = targets.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, targets)

            output_all.append([(o.cpu().detach().numpy(), t.cpu().detach().numpy()) for o,t in zip(output, targets)])

            # measure accuracy and record loss
            acc1, acc2 = accuracy(output, targets, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top2.update(acc2[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top2=top2))

        print(' * Acc@1 {top1.avg:.3f} Acc@2 {top2.avg:.3f}'
                  .format(top1=top1, top2=top2))

    return top1.avg, top2.avg, losses.avg, output_all


def save_checkpoint(state, is_best, folder, filename):
    torch.save(state, os.path.join(folder, filename))
    # if is_best:
    #     shutil.copyfile(os.path.join(folder, filename), os.path.join(folder, 'model_best.pth'))
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    global args

    main_arg_parser = argparse.ArgumentParser(description="parser for fast-neural-style")
    subparsers = main_arg_parser.add_subparsers(title="subcommands", dest="subcommand")

    train_arg_parser = subparsers.add_parser("train", help="parser for training arguments")
    train_arg_parser.add_argument("--cuda", type=int, default=1, help="cuda available or not")
    train_arg_parser.add_argument("--gpu", type=str, default='0,1,2,3', required=True, help="gpu")
    train_arg_parser.add_argument("--region", type=str, default='AC', required=True, help="eye location")
    train_arg_parser.add_argument("--arch_name", type=str, default='resnet50', required=True, help="model architecture")
    train_arg_parser.add_argument("--epochs", type=int, default=100, help="number of training epochs, default is 100")
    train_arg_parser.add_argument("--batch-size", type=int, default=32, help="batch size for training, default is 32")
    train_arg_parser.add_argument("--lr", type=float, default=1e-5, help="learning rate, default is 1e-5")
    train_arg_parser.add_argument("--seed", type=int, default=42, help="random seed for training")
    train_arg_parser.add_argument("--log-interval", type=int, default=10, help="number of images after which the training loss is logged, default is 10")
    train_arg_parser.add_argument("--lr_scheduler", type=int, default=0, help="1=use lr scheduler with ADAM")
    train_arg_parser.add_argument("--cls_weighted", type=int, default=0, help="1=class weighted")
    train_arg_parser.add_argument("--cutMix", type=bool, default=False, help="cutmix augmentation")
    train_arg_parser.add_argument("--mixup", type=bool, default=False, help="mixup augmentation")

    train_arg_parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    train_arg_parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    train_arg_parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    eval_arg_parser = subparsers.add_parser("eval", help="parser for evaluation/stylizing arguments")
    eval_arg_parser.add_argument("--cuda", type=int, required=True, help="set it to 1 for running on GPU, 0 for CPU")
    eval_arg_parser.add_argument("--gpu", type=str, default='0,1,2,3', help="gpu")
    eval_arg_parser.add_argument("--region", type=str, default='AC', required=True, help="eye location")
    eval_arg_parser.add_argument("--dset", type=str, default='test', required=True, help="eval dataset")
    eval_arg_parser.add_argument("--arch_name", type=str, default='resnet50', required=True, help="model architecture")
    eval_arg_parser.add_argument("--lr", type=float, default=1e-5, help="learning rate, default is 1e-5")
    eval_arg_parser.add_argument("--batch-size", type=int, default=32, help="batch size for training, default is 32")
    eval_arg_parser.add_argument('--snapshot', default='', type=str, metavar='PATH', help='path to prediction checkpoint (default: none)')
    eval_arg_parser.add_argument("--cls_weighted", type=int, default=0, help="1=class weighted")
    eval_arg_parser.add_argument("--log-interval", type=int, default=1000, help="number of images after which the training loss is logged")

    args = main_arg_parser.parse_args()

    if args.subcommand is None:
        print("ERROR: specify either train or eval")
        sys.exit(1)
    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    if args.subcommand == "train":
        train_wrapper()
    else:
        predict()

    return


# take bunch of images, predict and store results
def predict(runs=range(10)):
    from PeppleDataset import PeppleDataset, get_imgs_for_split

    # get weights
    weights_folder = os.path.join(CLASSIFICATION_FOLDER, 'torch_runs')
    pattern = '{}_lr{}'.format(args.arch_name, args.lr)
    if not args.snapshot:
        # epoch, best_acc, snapshot = find_best_weights(weight_folder=weights_folder, pattern=pattern)
        epoch, best_acc, snapshot = load_best_weights(weights_folder, pattern)
    else:
        snapshot = args.snapshot
    if sys.platform == "win32":
        snapshot = snapshot.replace('/data/yue/', 'z:/yue/')

    net = get_model_with_weights(snapshot, gpu=args.gpu)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available() and args.gpu is not None:
        # net = net.cuda(args.gpu)
        # criterion = criterion.cuda(args.gpu)
        print('cuda available and gpu={}'.format(args.gpu))
        net = net.cuda()
        criterion = criterion.cuda()
    else:
        1
        # net = net.cuda()
        # criterion = torch.nn.CrossEntropyLoss()

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    train_img_labels, valid_img_labels, test_img_labels = get_imgs_for_split(args.region, nrow=IMG_ROW, ncol=IMG_SIZE)

    if args.dset=='valid':
        val_dataset = PeppleDataset(valid_img_labels, transform=val_transform)
    elif args.dset=='test':
        val_dataset = PeppleDataset(test_img_labels, transform=val_transform)
    else:
        val_dataset = PeppleDataset(train_img_labels, transform=val_transform)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    top1_avg, top2_avg, losses_avg, output_all = validate(val_loader, net, criterion)
    # print(output_all)
    log_file = os.path.join(CLASSIFICATION_FOLDER, 'preds_{}_{}_w{}.csv'.format(args.region, args.dset, args.cls_weighted))
    with open(log_file, 'w') as fout:
        for batch_o in output_all:
            for o in batch_o:
                probs= o[0]
                target = o[1]
                print(probs, target)
                fout.write('{},{}\n'.format(','.join([str(x) for x in probs]), str(target)))
    fout.close()

    return


##
def find_all_best_weights(weight_folder):
    best_acc_snapshot_dict = {}
    for model_name in ['deepCNet1', 'deepCNet2', 'deepCNet3', 'deepCNet2_bn',]:
        for data_type in ['raw', 'segmented', 'combo']:
            for img_size in [320, 512]:
                for eye_radius in [500]:
                    for lr in [0.00001]:
                        pattern = '{}_s{}_r{}_d{}_lr{}'.format(model_name, img_size, eye_radius, data_type, lr)
                        epoch,  best_acc, snapshot_best = find_best_weights(weight_folder, pattern)
                        best_acc_snapshot_dict[pattern] = (epoch,  best_acc, snapshot_best)

    weights_json = os.path.join(CLASSIFICATION_FOLDER, 'torch_runs', 'best_weights.json')
    with open(weights_json, 'w') as fout:
        json.dump(best_acc_snapshot_dict, fout)
    fout.close()
    return


def load_best_weights(weight_folder, pattern):
    weights_json = os.path.join(CLASSIFICATION_FOLDER, 'torch_runs', 'best_weights.json')
    fin = open(weights_json).read()
    best_acc_snapshot_dict = json.loads(fin)

    if pattern in best_acc_snapshot_dict:
        return best_acc_snapshot_dict[pattern]
    else:
        return find_best_weights(weight_folder, pattern)


def find_best_weights(weight_folder, pattern, gpu=0):
    snapshots = glob.glob(os.path.join(weight_folder, '*{}*'.format(pattern)))
    snapshots = [x for x in snapshots if '.txt' not in x]
    # maintain epoch order
    snapshots = [os.path.join(weight_folder, '{}_{}'.format(pattern, idx)) for idx in range(len(snapshots))]
    epoch_acc = {}
    best_acc = 0
    best_epoch = -1
    for idx, snapshot in enumerate(snapshots):
        print(idx, snapshot)
        state_dict = load_weights(snapshot, gpu)
        state_epoch = state_dict['epoch']
        state_best_acc = state_dict['best_acc1']
        epoch_acc[state_epoch] = (state_epoch, float(state_best_acc), snapshot)
        if float(state_best_acc)>best_acc:
            best_acc = state_best_acc
            best_epoch=state_epoch
    if best_epoch in epoch_acc:
        return epoch_acc[best_epoch]
    else:
        return -1, 0, None


def get_model_with_weights(snapshot, gpu=0):
    net = get_model(args.arch_name, num_channels=3, num_classes=AC_CLASSES, input_rows=IMG_ROW, input_cols=IMG_SIZE)
    if snapshot is not None:
        state_dict = load_weights(snapshot, gpu)
        # net.load_state_dict(state_dict['state_dict'])
        state_dict_fixed = fix_state_dict_hack(state_dict['state_dict'])
        # for key, val in state_dict_fixed.items():
        #     print(key, val.shape)
        net.load_state_dict(state_dict_fixed)
        print("Snapshot for epoch {} loaded from {}".format(state_dict['epoch'], snapshot))
    return net


def fix_state_dict_hack(state_dict):
    state_dict_fixed = {}
    for key, val in state_dict.items():
        state_dict_fixed[key.replace('module.', '')] = val
    return state_dict_fixed


def load_weights(snapshot, gpu=0):
    if torch.cuda.is_available() and gpu is not None:  # cuda specified
        # state_dict = torch.load(snapshot, map_location="cuda:{}".format(gpu))
        state_dict = torch.load(snapshot)
    else:
        state_dict = torch.load(snapshot, map_location=lambda storage, loc: storage)
    return state_dict


class fake_args:
    def __init__(self, cuda=1, gpu=None, arch_name='resnet50', lr=0.0001, batch_size=32, snapshot='',):
        self.cuda = cuda
        self.gpu = gpu
        self.arch_name = arch_name
        self.lr = lr
        self.batch_size = batch_size
        self.snapshot = snapshot
        self.log_interval =10


if __name__ == "__main__":
    main()    # for training - see command_logs for experiments setups
    sys.exit()