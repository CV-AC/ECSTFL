import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision
from dataloader import Imgs_Dataset
from dataloader import Videos_Dataset
import matplotlib
matplotlib.use('Agg')
import datetime
import sklearn.metrics
import glob

now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H-%M]-")

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('-b', '--batch-size', default=32, type=int, metavar='N')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--list_file_val', type=str, default='./annotation/Imgs_test.csv', help='path of annotation file (.csv)')
parser.add_argument('--model_name', type=str, default="resnet18_ECSTFL_rafdb", help="resne18_rafdb, resnet18_ECSTFL_rafdb, r3d_dfew, r3d_ECSTFL_dfew")
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--type', type=str, default='img', help="img or video")


args = parser.parse_args()
now = datetime.datetime.now()
time_str = now.strftime("[%m-%d]-[%H_%M_%S]-")
log_txt_path = './log/{type}_test-{time_str}-log.txt'.format(type=args.type, time_str=time_str)
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


def main():

    # Bulid model and load pretraining parameters
    if args.device == 'gpu':
        if args.model_name == "resnet18_rafdb":
            model = torchvision.models.resnet18(num_classes=7, pretrained=False)
            model_weights_path = './checkpoint_load/resnet18_rafdb.pth'
            model_state = torch.load(model_weights_path)
            model.load_state_dict(model_state['state_dict'])
            model.cuda()
        if args.model_name == "resnet18_ECSTFL_rafdb":
            model = torchvision.models.resnet18(num_classes=7, pretrained=False)
            model_weights_path = './checkpoint_load/resnet18_ECSTFL_rafdb.pth'
            model_state = torch.load(model_weights_path)
            model.load_state_dict(model_state['state_dict'])
            model.cuda()
        if args.model_name == 'r3d_dfew':
            model = torchvision.models.video.resnet.r3d_18(num_classes=7, pretrained=False)
            model_weights_path = './checkpoint_load/r3d_dfew.pth'
            model_weights = torch.load(model_weights_path)['state_dict']
            model.load_state_dict(model_weights)
            model.cuda()
        if args.model_name == 'r3d_ECSTFL_dfew':
            model = torchvision.models.video.resnet.r3d_18(num_classes=7, pretrained=False)
            model_weights_path = './checkpoint_load/r3d_ECSTFL_dfew.pth'
            model_weights = torch.load(model_weights_path)['state_dict']
            model.load_state_dict(model_weights)
            model.cuda()
        criterion = nn.CrossEntropyLoss().cuda()

    if args.device == 'cpu':
        if args.model_name == "resnet18_rafdb":
            model = torchvision.models.resnet18(num_classes=7, pretrained=False)
            model_weights_path = './checkpoint_load/resnet18_rafdb.pth'
            model_state = torch.load(model_weights_path, map_location='cpu')
            model.load_state_dict(model_state['state_dict'])
        if args.model_name == "resnet18_ECSTFL_rafdb":
            model = torchvision.models.resnet18(num_classes=7, pretrained=False)
            model_weights_path = './checkpoint_load/resnet18_ECSTFL_rafdb.pth'
            model_state = torch.load(model_weights_path, map_location='cpu')
            model.load_state_dict(model_state['state_dict'])
        if args.model_name == 'r3d_dfew':
            model = torchvision.models.video.resnet.r3d_18(num_classes=7, weights=False)
            model_weights_path = './checkpoint_load/r3d_dfew.pth'
            model_weights = torch.load(model_weights_path)['state_dict']
            model.load_state_dict(model_weights)
        if args.model_name == 'r3d_ECSTFL_dfew':
            model = torchvision.models.video.resnet.r3d_18(num_classes=7, weights=False)
            model_weights_path = './checkpoint_load/r3d_ECSTFL_dfew.pth'
            model_weights = torch.load(model_weights_path)['state_dict']
            model.load_state_dict(model_weights)
        criterion = nn.CrossEntropyLoss()


    # Dataset and Dataloader
    if args.list_file_val is None and args.type == 'img':
        args.list_file_val = './annotation_infer/Imgs_test.csv'
        test_data = Imgs_Dataset.test_data_loader(args)
    if args.list_file_val is not None and args.type == 'img':
        test_data = Imgs_Dataset.test_data_loader(args)
    if args.list_file_val is None and args.type == 'video':
        args.list_file_val = './annotation_infer/Videos_test.txt'
        test_data = Videos_Dataset.test_data_loader(args)
    if args.list_file_val is not None and args.type == 'video':
        test_data = Videos_Dataset.test_data_loader(args)


    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    # inference
    validate(test_loader, model, criterion, args)



def validate(val_loader, model, criterion, args):
    losses = AverageMeter('Loss', ':.4f')
    top1 = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader),
                             [losses, top1],
                             prefix='Test: ')

    pres_te, trues_te = [], []
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (images, cate_label) in enumerate(val_loader):
            if args.device == 'gpu':
                images = images.cuda()
                cate_label = cate_label.cuda()
                if args.model_name in ['r3d_dfew', 'r3d_ECSTFL_dfew']: images = images.permute(0, 2, 1, 3, 4)

            if args.device == 'cpu':
                images = images
                cate_label = cate_label
                if args.model_name in ['r3d_dfew', 'r3d_ECSTFL_dfew']: images = images.permute(0, 2, 1, 3, 4)

            # compute output
            out = model(images)
            loss = criterion(out, cate_label)

            # measure accuracy and record loss
            acc1, _ = accuracy(out, cate_label, topk=(1, 5))
            # This loss is Cross-Entropy loss. It is NOT CE+ECSTFL.
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            _, pre_te = torch.max(out, 1)
            pres_te += pre_te.cpu().numpy().tolist()
            trues_te += cate_label.cpu().numpy().tolist()
            UAR_te = get_UAR(trues_te, pres_te)

            # print loss and accuracy
            if i % args.print_freq == 0:
                progress.display(i)

        print('Current Accuracy: {top1.avg:.3f}, UAR: {UAR_te:.3f}'.format(top1=top1, UAR_te=UAR_te))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {top1.avg:.3f}, UAR: {UAR_te:.3f}'.format(top1=top1, UAR_te=UAR_te) + '\n')

    return top1.avg, UAR_te, losses.avg, pres_te, trues_te




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
