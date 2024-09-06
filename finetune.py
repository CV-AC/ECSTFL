import argparse
import time
import shutil
import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import datetime
from dataloader import Imgs_Dataset
from dataloader import Videos_Dataset
import ECSTFL
import glob
import sklearn

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, metavar='W', dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint')
parser.add_argument('-sss','--scheduler_stepsize', type=int, default=50)

parser.add_argument('--list_file_val', type=str, default='./annotation/Imgs_test.csv', help='path of annotation file (.csv)')
parser.add_argument('--list_file_train', type=str, default='./annotation/Img_test.csv', help='path of annotation file (img .csv; video .txt)')
parser.add_argument('--model_name', type=str, default="resnet18_ECSTFL_rafdb", help="resnet18_rafdb, resnet18_ECSTFL_rafdb, r3d_dfew, r3d_ECSTFL_dfew")
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--type', type=str, default='img', help="img or video")
parser.add_argument('--gamma', type=float, default=0, help='a float number')

args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H_%M_%S]-")
log_txt_path = './log/{type}_finetune-{time_str}-log.txt'.format(type=args.type, time_str=time_str)
log_curve_path = './log/{type}_finetune-{time_str}-log.png'.format(type=args.type, time_str=time_str)
checkpoint_path = './checkpoint/{type}_finetune-{time_str}-model.pth'.format(type=args.type, time_str=time_str)
best_checkpoint_path = './checkpoint/{type}_finetune-{time_str}-model_best.pth'.format(type=args.type, time_str=time_str)
softmax = nn.Softmax(dim=1)

def get_WAR(trues_te, pres_te):
    WAR  = sklearn.metrics.accuracy_score(trues_te, pres_te)
    return WAR

def get_UAR(trues_te, pres_te):
    cm = sklearn.metrics.confusion_matrix(trues_te, pres_te)
    acc_per_cls = [ cm[i,i]/sum(cm[i]) for i in range(len(cm))]
    UAR = sum(acc_per_cls)/len(acc_per_cls)*100
    return UAR

def get_cm(trues_te, pres_te):
    cm = sklearn.metrics.confusion_matrix(trues_te, pres_te)
    return cm

features = {}
def get_features(name):
    # to hook the output of specific layer
    def hook(model, input, output):
        features[name] = output
    return hook



def main():
    best_acc = 0
    recorder = RecorderMeter(args.epochs)
    print("list_file_train:{list_file_train}".format(list_file_train=args.list_file_train))
    print("list_file_val:{list_file_val}".format(list_file_val=args.list_file_val))

    print('The training time: ' + now.strftime("%m-%d %H:%M"))
    with open(log_txt_path, 'a') as f:
        f.write('The list_file_train:{list_file_train}; The list_file_val:{list_file_val}.'.format(list_file_train=args.list_file_train, list_file_val=args.list_file_val))

    # Bulid model, load pretraining parameters; define loss function
    if args.device == 'gpu':
        if args.model_name == "resnet18":
            model = torchvision.models.resnet18(pretrained=False, num_classes=7)
            model.cuda()
        if args.model_name == "resnet18_ECSTFL":
            model = torchvision.models.resnet18(pretrained=False, num_classes=7)
            model.cuda()
        if args.model_name == "resnet18_rafdb":
            model = torchvision.models.resnet18(num_classes=7)
            model_weights_path = './checkpoint_load/resnet18_rafdb.pth'
            model_state = torch.load(model_weights_path)
            model.load_state_dict(model_state['state_dict'])
            model.cuda()
        if args.model_name == "resnet18_ECSTFL_rafdb":
            model = torchvision.models.resnet18(num_classes=7)
            model_weights_path = './checkpoint_load/resnet18_ECSTFL_rafdb.pth'
            model_state = torch.load(model_weights_path)
            model.load_state_dict(model_state['state_dict'])
            model.cuda()
        if args.model_name == 'r3d':
            model = torchvision.models.video.resnet.r3d_18(weights=False, num_classes=7)
            model.cuda()
        if args.model_name == 'r3d_ECSTFL':
            model = torchvision.models.video.resnet.r3d_18(weights=False, num_classes=7)
            model.cuda()
        if args.model_name == 'r3d_dfew':
            model = torchvision.models.video.resnet.r3d_18(weights=False, num_classes=7)
            model_weights_path = './checkpoint_load/r3d_dfew.pth'
            model_weights = torch.load(model_weights_path)['state_dict']
            model.load_state_dict(model_weights)
            model.cuda()
        if args.model_name == 'r3d_ECSTFL_dfew':
            model = torchvision.models.video.resnet.r3d_18(weights=False, num_classes=7)
            model_weights_path = './checkpoint_load/r3d_ECSTL_dfew.pth'
            model_weights = torch.load(model_weights_path)['state_dict']
            model.load_state_dict(model_weights)
            model.cuda()
        criterion_CE = nn.CrossEntropyLoss().cuda()
        criterion_ECSTFL = ECSTFL.ECSTFL().cuda()

    if args.device == 'cpu':
        if args.model_name == "resnet18":
            model = torchvision.models.resnet18(pretrained=False, num_classes=7)
        if args.model_name == "resnet18_ECSTFL":
            model = torchvision.models.resnet18(pretrained=False, num_classes=7)
        if args.model_name == "resnet18_rafdb":
            model = torchvision.models.resnet18(num_classes=7)
            model_weights_path = './checkpoint_load/resnet18_rafdb.pth'
            model_state = torch.load(model_weights_path)
            model.load_state_dict(model_state['state_dict'])
        if args.model_name == "resnet18_ECSTFL_rafdb":
            model = torchvision.models.resnet18(num_classes=7)
            model_weights_path = './checkpoint_load/resnet18_ECSTFL_rafdb.pth'
            model_state = torch.load(model_weights_path)
            model.load_state_dict(model_state['state_dict'])
        if args.model_name == 'r3d':
            model = torchvision.models.video.resnet.r3d_18(weights=False, num_classes=7)
        if args.model_name == 'r3d_ECSTFL':
            model = torchvision.models.video.resnet.r3d_18(weights=False, num_classes=7)
        if args.model_name == 'r3d_dfew':
            model = torchvision.models.video.resnet.r3d_18(weights=False, num_classes=7)
            model_weights_path = './checkpoint_load/r3d_dfew.pth'
            model_weights = torch.load(model_weights_path)['state_dict']
            model.load_state_dict(model_weights)
        if args.model_name == 'r3d_ECSTFL_dfew':
            model = torchvision.models.video.resnet.r3d_18(weights=False, num_classes=7)
            model_weights_path = './checkpoint_load/r3d_ECSTL_dfew.pth'
            model_weights = torch.load(model_weights_path)['state_dict']
            model.load_state_dict(model_weights)
        criterion_CE = nn.CrossEntropyLoss()
        criterion_ECSTFL = ECSTFL.ECSTFL()


    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_stepsize, gamma=0.1)

    # Registering hook
    hookinglayer_name = 'ECSTFL_features'
    if args.model_name in ['resnet18_ECSTFL', 'resnet18_ECSTFL_rafdb']: model.avgpool.register_forward_hook(get_features(hookinglayer_name))
    if args.model_name in ['r3d_ECSTFL', 'r3d_ECSTFL_dfew']: model.avgpool.register_forward_hook(get_features(hookinglayer_name))


    # Dataset and Dataloader
    if args.type == 'img':
        #args.list_file_train = './annotataon/Imgs_train.csv'
        #args.list_file_val = './annotation/Imgs_test.csv'
        train_data = Imgs_Dataset.train_data_loader(args)
        test_data = Imgs_Dataset.test_data_loader(args)
    if args.type == 'video':
        #args.list_file_train = './annotation/Videos_train.txt'
        #args.list_file_val = './annotation/Videos_test.txt'
        train_data = Videos_Dataset.train_data_loader(args)
        test_data = Videos_Dataset.test_data_loader(args)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    # finetuning all layers of networks
    for epoch in range(args.epochs):
        inf = '********************' + str(epoch) + '********************'
        start_time = time.time()
        current_learning_rate = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path, 'a') as f:
            f.write(inf + '\n')
            f.write('Current learning rate: ' + str(current_learning_rate) + '\n')

        print(inf)
        print('Current learning rate: ', current_learning_rate)

        # train & test
        train_acc, train_los = train(train_loader, model, criterion_CE, criterion_ECSTFL, features, optimizer, epoch, args)
        val_acc, val_los = validate(test_loader, model, criterion_CE, criterion_ECSTFL, features, args)
        scheduler.step()

        # remember best acc and save checkpoint
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({'epoch': epoch + 1,
                         'state_dict': model.state_dict(),
                         'best_acc': best_acc,
                         'optimizer': optimizer.state_dict(),
                         'recorder': recorder}, is_best)

        # print and save log
        epoch_time = time.time() - start_time
        recorder.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder.plot_curve(log_curve_path)

        print('The best accuracy: {:.3f}'.format(best_acc.item()))
        print('An epoch time: {:.1f}s'.format(epoch_time))
        with open(log_txt_path, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc.item()) + '\n')
            f.write('An epoch time: {:.1f}s' + str(epoch_time) + '\n')


def train(train_loader, model, criterion_CE, criterion_ECSTFL, features, optimizer, epoch, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader),
                             [losses, top1],
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    for i, (images, target) in enumerate(train_loader):
        if args.device == 'gpu':
            images = images.cuda()
            target = target.cuda()
        if args.model_name in ['r3d', 'r3d_ECSTFL','r3d_dfew', 'r3d_ECSTFL_dfew']:
            images = images.permute(0, 2, 1, 3, 4)

        out = model(images)

        if args.model_name in ['resnet18', 'resnet18_rafdb', 'r3d', 'r3d_dfew']:
            loss_CE = criterion_CE(out, target)
            loss = loss_CE
        if args.model_name in ['resnet18_ECSTFL', 'resnet18_ECSTFL_rafdb', 'r3d_ECSTFL', 'r3d_ECSTFL_dfew']:
            loss_CE = criterion_CE(out, target)
            loss_ECSTFL = criterion_ECSTFL(features['ECSTFL_features'].squeeze(), target)
            loss = loss_CE + float(args.gamma)*loss_ECSTFL

        # measure accuracy and record loss
        acc1, _ = accuracy(out, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion_CE, criterion_ECSTFL, features, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if args.device == 'gpu':
                images = images.cuda()
                target = target.cuda()
            if args.model_name in ['r3d', 'r3d_ECSTFL', 'r3d_dfew', 'r3d_ECSTFL_dfew']:
                images = images.permute(0, 2, 1, 3, 4)

            out = model(images).detach()

            if args.model_name in ['resnet18', 'resnet18_rafdb', 'r3d', 'r3d_dfew']:
                loss_CE = criterion_CE(out, target)
                loss = loss_CE
            if args.model_name in ['resnet18_ECSTFL', 'resnet18_ECSTFL_rafdb', 'r3d_ECSTFL', 'r3d_ECSTFL_dfew']:
                loss_CE = criterion_CE(out, target)
                loss_ECSTFL = criterion_ECSTFL(features['ECSTFL_features'].squeeze(), target)
                loss = loss_CE + float(args.gamma) * loss_ECSTFL

            # measure accuracy and record loss
            acc1, _ = accuracy(out, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print('Current Accuracy: {top1.avg:.3f}'.format(top1=top1))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}'.format(top1=top1) + '\n')
    return top1.avg, losses.avg


def save_checkpoint(state, is_best):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':
    main()






    3

