import argparse
import os
import shutil
import time
import re

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from neural_style.transformer_net import TransformerNet
from neural_style.small_transformer_net import SmallTransformerNet
import vgg
from vgg_bis import Vgg16
import utils
import slim


model_names = sorted(name for name in vgg.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg19)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--teacher-checkpoint', default='', type=str, metavar='PATH',
                    help='Path to the checkpoint of the teacher (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--slim-checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest slim checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)

parser.add_argument('--coco', action='store_true',
                    help='use coco dataset ')
parser.add_argument('--coco-dataset', type=str,
                    help='path to coco dataset ')
parser.add_argument("--style-image", type=str, default="images/style-images/mosaic.jpg",
                    help="path to style-image")
parser.add_argument("--style-size", type=int, default=None,
                    help="size of style-image, default is the original size of style image")
parser.add_argument("--content-weight", type=float, default=1e5,
                    help="weight for content-loss, default is 1e5")
parser.add_argument("--style-weight", type=float, default=1e10,
                    help="weight for style-loss, default is 1e10")

best_prec1 = 0
best_loss = 0

def get_train_loader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ])

    if args.coco:
        data = datasets.ImageFolder(args.coco_dataset, transform=transform)
        print("COCO loaded")
    else:
        data = datasets.CIFAR10(root='./data', train=True, transform=transform,
                                download=True)

    return torch.utils.data.DataLoader(
        data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)


def get_val_loader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),download=True),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


def get_vgg_model(args):
    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    model = vgg.__dict__[args.arch]()

    model.features = torch.nn.DataParallel(model.features)
    model.cuda()

    # optionally resume from a checkpoint
    if args.teacher_checkpoint:
        if os.path.isfile(args.teacher_checkpoint):
            print("=> loading checkpoint '{}'".format(args.teacher_checkpoint))
            checkpoint = torch.load(args.teacher_checkpoint)
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.teacher_checkpoint))

    return model


def get_style_network(args):
    """ Get style network
    """
    with torch.no_grad():
        style_model = TransformerNet()
        style_model.cuda()

        if args.teacher_checkpoint:
            if os.path.isfile(args.teacher_checkpoint):
                #load_weight(style_model, args.teacher_checkpoint)
                state_dict = torch.load(args.teacher_checkpoint)
                # remove saved deprecated running_* keys in InstanceNorm from the checkpoint
                for k in list(state_dict.keys()):
                    if re.search(r'in\d+\.running_(mean|var)$', k):
                        del state_dict[k]
                style_model.load_state_dict(state_dict)

                print("Loaded checkpoint for teacher network")
            else:
                print("=> no checkpoint found at '{}'".format(args.teacher_checkpoint))

    return style_model


def get_slim_model(args):
    """Get slim model
    """
    slim_model = SmallTransformerNet()
    slim_model.cuda()

    return slim_model


def load_weight(model, checkpoint_path):
    """Load weights into model
    """
    if os.path.isfile(checkpoint_path):
        print"=> loading checkpoint '{}'".format(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.evaluate, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(checkpoint_path))


def distillation_loss(y, labels, teacher_scores, T, alpha):
    return (
        nn.KLDivLoss()(nn.functional.log_softmax(y / T), nn.functional.softmax(teacher_scores/T)) 
        * (T*T * 2.0 * alpha)
        #+ nn.functional.cross_entropy(y, labels) * (1. - alpha)
        )


def style_distillation_loss(output_teacher1, output_teacher2, output_teacher3, output_student1, output_student2, output_student3):
    mse_loss = torch.nn.MSELoss()
    loss_style1 = mse_loss(output_teacher1, output_student1)/float(10**0)
    loss_style2 = mse_loss(output_teacher2, output_student2)/float(10**1)
    loss_style3 = mse_loss(output_teacher3, output_student3)/float(10**5)        
    return 0*loss_style1 + 0*loss_style2 + loss_style3
    #return torch.norm(output_teacher - output_student) / float(10**5)
    #mse_loss = torch.nn.MSELoss()
    #return mse_loss(output_teacher, output_student) / float(10**6)



"""
MAIN
"""
def main():
    global args, best_prec1, best_loss
    args = parser.parse_args()

    device = "cuda"
    vgg_bis = Vgg16(requires_grad=False).to(device)
    style_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    
    style = utils.load_image(args.style_image, size=args.style_size)
    style = style_transform(style)
    style = style.repeat(args.batch_size, 1, 1, 1).to(device)
    
    features_style = vgg_bis(utils.normalize_batch(style))
    gram_style = [utils.gram_matrix(y) for y in features_style]

    big_model = get_style_network(args)

    """small_model = slim.SimpleConvNet(hidden=1000)
    small_model.cuda()
    if small_model is slim.DeepConvNet:
        for m in small_model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.normal_(0, 0.0)"""
    small_model = get_slim_model(args)

    cudnn.benchmark = True
    train_loader = get_train_loader(args)
    val_loader = get_val_loader(args)

    # define loss function (criterion) and optimizer
    criterion = style_distillation_loss

    if args.half:
        small_model.half()
        criterion.half()

    optimizer = torch.optim.SGD(small_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    mse_loss = torch.nn.MSELoss()

    if args.evaluate:
        if args.slim_checkpoint:
            load_weight(small_model, args.slim_checkpoint)
            validate(val_loader, small_model, criterion)
        else:
            print("Error: slim checkpoint path not set")
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        loss = train_distill(train_loader, big_model, small_model, criterion, optimizer, epoch, vgg_bis, gram_style)

        # evaluate on validation set
        #prec1 = validate(val_loader, small_model, criterion)

        # compute the loss on validation set
        #loss = validate_loss(val_loader, big_model, small_model, criterion)

        # remember the best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(best_loss, loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': small_model.state_dict(),
            'best_loss': best_loss,
        }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

        # remember best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1
        #best_prec1 = max(prec1, best_prec1)
        #save_checkpoint({
        #    'epoch': epoch + 1,
        #    'state_dict': small_model.state_dict(),
        #    'best_prec1': best_prec1,
        #}, is_best, filename=os.path.join(args.save_dir, 'checkpoint_{}.tar'.format(epoch)))

    print("Best loss: {}".format(best_loss))


def train_distill(train_loader, big_model, small_model, criterion, optimizer, epoch, vgg_bis, gram_style):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    small_model.train()
    big_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
	
	n_batch = len(input)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        #target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute big model output
        with torch.no_grad():
            teacher_output1 = big_model(input_var).relu_1.data.cpu().numpy()
            teacher_output1 = torch.cuda.FloatTensor(teacher_output1)
            teacher_output2 = big_model(input_var).relu_2.data.cpu().numpy()
            teacher_output2 = torch.cuda.FloatTensor(teacher_output2)
            teacher_output3 = big_model(input_var).relu_3.data.cpu().numpy()
            teacher_output3 = torch.cuda.FloatTensor(teacher_output3)

        # compute output
        output1 = small_model(input_var).relu_1
        output2 = small_model(input_var).relu_2
        output3 = small_model(input_var).relu_3


	    # convert original target to one-hot representation
        #target_onehot = torch.cuda.FloatTensor(*output.size())
        #target_onehot.zero_()
        #target_onehot.scatter_(1, target_var.view(-1, 1), 1)
        mse_loss = torch.nn.MSELoss()  
        y = utils.normalize_batch(output3)
        x = utils.normalize_batch(input_var)
        
        features_y = vgg_bis(y)
        features_x = vgg_bis(x)

        content_loss = args.content_weight * mse_loss(features_y.relu2_2, features_x.relu2_2)
	content_loss /=(1.0*float(10**5)) 

        style_loss = 0.
        for ft_y, gm_s in zip(features_y, gram_style):
            gm_y = utils.gram_matrix(ft_y)
	    # print('gm_y = ',gm_y)
            style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :]) #style_loss += mse_loss(gm_y, gm_s[:n_batch, :, :])
        style_loss *= args.style_weight
	style_loss /= (3.0*float(10**8))

        #loss = criterion(output, target_var, teacher_output, T=20.0, alpha=0.7)
        loss = criterion(teacher_output1, teacher_output2, teacher_output3, output1, output2, output3)     
	loss1 = mse_loss(teacher_output1, output1)/float(10**0)
        loss2 = mse_loss(teacher_output2, output2)/float(10**1)
        loss3 = mse_loss(teacher_output3, output3)/float(10**5)
        
        # loss = loss_norm +0*content_loss +0*style_loss
	# loss /= float(10**3)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # output3 = output3.float()
        loss = loss.float()
	loss1 = loss1.float()
	loss2 = loss2.float()
	loss3 = loss3.float()
        losses.update(loss.data.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))
	#    print('loss1 = ',loss1,'loss2 = ',loss2,'loss3 = ',loss3)
	#    print('loss1 {loss1.item()} ({loss1.item()})\t'
	#	  'loss2 {loss2.item()} ({loss2.item()})\t'
	#	  'loss3 {loss3.item()} ({loss3.item()})'.format(
	#	      loss1=loss1,loss2=loss2,loss3=loss3))
	    print('loss1 = ',loss1.item(),'loss2 = ',loss2.item(),'loss3 = ',loss3.item())
            print('content_loss = ', content_loss.item(), 'style_loss = ',style_loss.item())
    return losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target)
        if args.half:
            input_var = input_var.half()

        # compute big model output
        with torch.no_grad():
            teacher_output = big_model(input_var).data.cpu().numpy()
            teacher_output = torch.cuda.FloatTensor(teacher_output)

        # compute output
        output = small_model(input_var)

	    # convert original target to one-hot representation
        #target_onehot = torch.cuda.FloatTensor(*output.size())
        #target_onehot.zero_()
        #target_onehot.scatter_(1, target_var.view(-1, 1), 1)

        #loss = criterion(output, target_var, teacher_output, T=20.0, alpha=0.7)
        loss = criterion(teacher_output, output)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        losses.update(loss.data.item(), input.size(0))

    return top1.avg


def validate_loss(val_loader, big_model, small_model, criterion):
    """ Compute an averaged loss for the validation dataset
    """
    losses = AverageMeter()

    # switch to train mode
    small_model.train()
    big_model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input_var = torch.autograd.Variable(input).cuda()

        if args.half:
            input_var = input_var.half()

        # compute big model output
        with torch.no_grad():
            teacher_output = big_model(input_var).data.cpu().numpy()
            teacher_output = torch.cuda.FloatTensor(teacher_output)

        # compute output
        with torch.no_grad():
            student_output = small_model(input_var)

        # Compute the loss
        loss = criterion(teacher_output, student_output).float()
        losses.update(loss.data.item(), input.size(0))

    return losses.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
