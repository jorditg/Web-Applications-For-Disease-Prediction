import subprocess

import cv2
import albumentations as A
import augmentations
from albumentations.pytorch import ToTensor

import sys
import argparse
import os
import shutil
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from PIL import Image

import models

#import sampler

from radam import RAdam

from torch.utils.tensorboard import SummaryWriter

import sklearn.metrics as sklm
import scipy as sp
from functools import partial

from sklearn.model_selection import KFold

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

parser = argparse.ArgumentParser(description='PyTorch Retine Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('-folds', type=int, metavar='N', help='number of cross-validation folds')
parser.add_argument('--val-fold', type=int, metavar='N', help='validation fold')
parser.add_argument('--seed', type=int, default=1234, metavar='N', help='cross validation random seed')
parser.add_argument('--arch', '-a', default='model_best.pth.tar', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 15)')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--mu', '--momentum', default=0.9, type=float,
                    metavar='MU', help='momentum (when used by optimizer, otherwise ignored)')
parser.add_argument('--weight-decay', '--wd', default=0.0, type=float,
                    metavar='W', help='weight decay (default: 0.0)')
parser.add_argument('--print-freq', '-p', default=250, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-c', '--classes', default=9, type=int,
                    metavar='N', help='number of classes (default: 5)')
parser.add_argument('--opt-method', default='RAdam', type=str,
                    help='Optimizer method: SGD, Adam, RAdam')                    
parser.add_argument('--nesterov', default='False', type=bool,
                    help='Nesterov acceleration when available')     
parser.add_argument('-s', '--input-size', default=224, type=int,
                    metavar='N', help='input image size')
parser.add_argument('--model-type', default='efficientnet-b0', type=str,
                    help='Model type ')               
parser.add_argument('--deterministic', action='store_true')
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--sync_bn', action='store_true',
                    help='enabling apex sync BN.')
parser.add_argument('--opt-level', type=str)
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

args = parser.parse_args()
writer = SummaryWriter() # Tensorboard writer

if __name__ == '__main__':
    main()

def main():
    global args

    best_prec1 = 0.0

    print("opt_level = {}".format(args.opt_level))
    print("keep_batchnorm_fp32 = {}".format(args.keep_batchnorm_fp32), type(args.keep_batchnorm_fp32))
    print("loss_scale = {}".format(args.loss_scale), type(args.loss_scale))

    print("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        checkpoint = torch.load(args.arch)
        model = models.load_model(args.model_type, args.input_size)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model:")
        model = models.load_model(args.model_type, args.input_size)

    if args.sync_bn:
        import apex
        print("using apex synced BN")
        model = apex.parallel.convert_syncbn_model(model)

    #model = models.regression_model(model) # puts a tanh with class scaling

    model = model.cuda()

    print(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # mean and stdev from ISIC dataset
    mean = [166.43850410293402 / 255.0, 133.5872994553671 / 255.0, 132.33856917079888 / 255.0]
    std = [59.679343313897874 / 255.0, 53.83690126788451 / 255.0, 56.618447349633676 / 255.0]   

    albumentations_transform = augmentations.get_retinopathy_augmentations((args.input_size, args.input_size), mean, std)
      
    val_transforms = transforms.Compose([
            transforms.Resize(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
    ])    
    
    # Data loading code
    traindir = os.path.join(args.data, '')
        
    df = pd.read_csv(traindir + "/ISIC_2019_Training_GroundTruth.csv")
    files = df['image'].values
    categories=df.columns[1:]
    labels = df.iloc[:,1:].idxmax(axis=1).astype("category", categories=categories).cat.codes.values
    
    # set the validation fold that you wanna use for training
    n_folds = args.folds
    random_state = args.seed
    val_fold = args.val_fold
    print(f"Training for {n_folds} folds. Validation fold: {val_fold}. Random state: {random_state}")
    

    # don't change the random state!
    kf = KFold(n_splits=n_folds, shuffle=True, random_state = random_state)
    
    for i, (train_index, test_index) in enumerate(kf.split(files)):
        X_train, X_test = files[train_index], files[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        if val_fold == i:
            break    
    
    dataset_train = AlbumentationsDataset(traindir, X_train, y_train, albumentations_transform)
    dataset_val = TorchvisionDataset(traindir, X_test, y_test, val_transforms)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle = True,
        #sampler = sampler.ImbalancedDatasetSampler(dataset_train),
        num_workers=args.workers,
        pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=1, 
        shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.LogSoftmax()

    optimizer = select_optimizer(args, model)


    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        # train for 1 epoch        
        train(train_loader, model, criterion, optimizer, epoch, categories)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, epoch, categories)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        filename = f"checkpoint_{epoch+1}_{prec1:.3f}.pth.tar"

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename)

    # export scalar data to JSON for external processing
    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()
    
def select_optimizer(args, model):
    if args.opt_method == "SGD":
        print("using SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mu, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.opt_method == "Adam":
        print("using Adam")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt_method == "RAdam":
        print("using RAdam")
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise Exception("Optimizer not supported")
    return optimizer    

def saveImageFromTensor(inputBatchTensor):
    print("saveImageFromTensor")
    trans = transforms.ToPILImage()
    for i in range(inputBatchTensor.size(0)):
        tensor = inputBatchTensor[i]
        img = trans(tensor)
        img.save("batch-" + str(i) + ".jpg")
        
def train(train_loader, model, criterion, optimizer, epoch, categories):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    y = np.ndarray((0), dtype='int64')
    t = np.ndarray((0), dtype='int64')
    end = time.time()
    df_pred = pd.DataFrame(columns=categories)
    df_true = pd.DataFrame(columns=categories)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #adjust_learning_rate(optimizer, epoch, i, len(train_loader))
        
        # With probability 1/nclasses switch target class to unknown class
        if random.random() < 0.111:
            target = 8 # unknown class

        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input.cuda())
        target_var = torch.autograd.Variable(target.unsqueeze(1).float().cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure qwk and record loss
        losses.update(loss.data, input.size(0))
        top1.update(loss.data, input.size(0))
        
        pred = output.data.cpu().numpy()
        true_val = target.cpu().numpy()

        df_pred.append(pd.DataFrame(pred, columns=categories), ignore_index=True)
        
        y = np.append(y, np.argmax(pred, axis=1))
        t = np.append(t, true_val)

        # compute gradient and do SGD step
        optimizer.zero_grad()

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'QWK {top1.val:.3f} ({top1.avg:.3f})')
            #saveImageFromTensor(input)
            #sys.exit()
    
    # as the training is randomized we rebuild the randomized version of the training for evaluation
    new_arr = np.zeros(shape=(t.shape[0],len(categories)))
    new_arr[np.arange(new_arr.shape[0]), t] = 1.0
    df_true.append(pd.DataFrame(new_arr, columns=categories), ignore_index=True)
    
    # just for evaluating them
    df_pred['image'] = df_pred.index.astype('string')
    df_true['image'] = df_pred.index.astype('string')

    pred_file = "pred.csv"
    true_file = "true.csv"

    df_pred.to_csv(pred_file, index=False)
    df_true.to_csv(true_file, index=False)

    bal_acc, outstr = isic_challenge_scoring(pred_file, true_file)

    print(outstr)
    print_confusion_matrix(y, t)
    
    writer.add_scalar(f'data/train_balacc', bal_acc, epoch)
    writer.add_scalar(f'data/train_loss', losses.avg, epoch)
    
    print(f'Epoch: [{epoch}]  * Train Balanced Accuracy {bal_acc}')


def validate(val_loader, model, criterion, epoch, categories):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    y = np.ndarray((0), dtype='int64')
    t = np.ndarray((0), dtype='int64')
    df_pred = pd.DataFrame(columns=categories)
    df_true = pd.DataFrame(columns=categories)
    for i, (input, target) in enumerate(val_loader):
        #target = target.cuda(async=True)
        with torch.no_grad():
            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.unsqueeze(1).float().cuda())

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        output = output.data.cpu().numpy()
        target = target.cpu().numpy()

        # measure qwk and record loss
        losses.update(loss.data, input.size(0))
        top1.update(loss.data, input.size(0))        

        pred = output.data.cpu().numpy()
        true_val = target.cpu().numpy()

        df_pred.append(pd.DataFrame(pred, columns=categories), ignore_index=True)
        
        y = np.append(y, np.argmax(pred, axis=1))
        t = np.append(t, true_val)
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time))

    # as the training is randomized we rebuild the randomized version of the training for evaluation
    new_arr = np.zeros(shape=(t.shape[0],len(categories)))
    new_arr[np.arange(new_arr.shape[0]), t] = 1.0
    df_true.append(pd.DataFrame(new_arr, columns=categories), ignore_index=True)
    
    # just for evaluating them
    df_pred['image'] = df_pred.index.astype('string')
    df_true['image'] = df_pred.index.astype('string')

    pred_file = "pred_val.csv"
    true_file = "true_val.csv"

    df_pred.to_csv(pred_file, index=False)
    df_true.to_csv(true_file, index=False)

    bal_acc, outstr = isic_challenge_scoring(pred_file, true_file)
    
    print(outstr)
    print_confusion_matrix(y, t)
    
    writer.add_scalar(f'data/val_balacc', bal_acc, epoch)
    writer.add_scalar(f'data/val_loss', losses.avg, epoch)
    
    print(f'Epoch: [{epoch}]  * Val Balanced Accuracy {bal_acc}')

    return bal_acc

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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
     
# Pretty print of confusion matrix
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, file=sys.stdout):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ", file=file)
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ", file=file)
    print(file=file)
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ", file=file)
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ", file=file)
        print(file=file)

def print_confusion_matrix(t, y):
    # t (true class number) and y (predicted class number) being arrays of shape (n_samples,)
    labels = [f'C{i}' for i in range(np.max(t))] 
    cnf_matrix = sklm.confusion_matrix(t, y)
    np.set_printoptions(precision=2)
    print_cm(cnf_matrix, labels=labels) 

def save_results_file(y, filename_pred = None, image_filenames = None):
    d = {'col1': ts1, 'col2': ts2}
    df = DataFrame(data=d, index=index)    


# required to install isic-challenge-scoring program using pip
def isic_challenge_scoring(pred_file, true_file = None):
    program = 'isic-challenge-scoring'
    
    if true_file == None:
        groundtruth_file = 'ISIC_2019_Training_GroundTruth.csv'
    
    output = subprocess.run([program, 
                             'classification', 
                             groundtruth_file, 
                             pred_file], stdout=subprocess.PIPE) 
    output = output.stdout.decode('utf-8')
    balanced_accuracy = float(output.split("balanced_accuracy")[1])
    return balanced_accuracy, output

class TorchvisionDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, filenames, labels, transform=None):
        self.file_path = file_path
        self.filenames = filenames
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.file_path + "/" + self.filenames[idx]
        
        # Read an image with PIL
        image = Image.open(file_path)
        if self.transform:
            image = self.transform(image)
        return image, label

class AlbumentationsDataset(torch.utils.data.Dataset):
    """__init__ and __len__ functions are the same as in TorchvisionDataset"""
    def __init__(self, traindir, file_paths, labels, transform=None):
        self.traindir = traindir        
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        file_path = self.traindir + "/" + self.file_paths[idx]
        
        # Read an image with OpenCV
        image = cv2.imread(file_path)
        
        # By default OpenCV uses BGR color space for color images,
        # so we need to convert the image to RGB color space.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, label

