import argparse
import math
import torch
import torch.nn 
import numpy as np
from cifar10 import *
from resnet import *
import torch.nn.functional as F
import torch.nn as nn
import random
from complementary_loss import *
import logging
from pcl import *
from torch.utils.tensorboard import SummaryWriter

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device

torch.set_printoptions(precision=2, sci_mode=False)

parser = argparse.ArgumentParser(description='PyTorch implementation')
parser.add_argument('--dataset', default='cifar10', type=str, 
                    choices=['cifar10', 'svhn'],
                    help='dataset name (cifar10)')
parser.add_argument('--epochs', default=500, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when ')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lam', default=1, type=float)
parser.add_argument('--proto_m', default=0.99, type=float)
parser.add_argument('--tau_proto', default=1, type=float)
args = parser.parse_args()

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

writer = SummaryWriter('runs/PCL_cifar10')

if args.dataset == 'cifar10':
    train_loader, test_loader = load_cifar10(batch_size=args.batch_size)
elif args.dataset == 'svhn':
    train_loader, test_loader = load_svhn(batch_size=args.batch_size)
else:
    raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

model = PCL(args, resnet18).to(device)
pc = (1.-F.one_hot(train_loader.dataset.given_label_matrix.to(int)))/9
# print(pc)

optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.05,
                                momentum=0.9,
                                weight_decay=1e-4)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[99, 199, 299], gamma=0.1, last_epoch= -1)


def train(train_loader, model, optimizer, epoch):
    
    losses = AverageMeter()
    losses_kl = AverageMeter()
    losses_cl = AverageMeter()
    losses_proto = AverageMeter()

    # define loss function
    kl_div = nn.KLDivLoss(reduction='batchmean').to(device)
    model.train()
    for i, (image_w, image_s, comp_y, y, idx) in enumerate(train_loader):
        image_w, image_s, comp_y, y = image_w.to(device), image_s.to(device), comp_y.to(device), y.to(device)
        logit_t_w, logit_t_s, logits_prot, logits_prot2, pseudo_label, logits_com, _ = model(image_w, image_s, comp_y)
        pred_t_w = F.softmax(logit_t_w, dim=1)
        pred_t_s = F.softmax(logit_t_s, dim=1)
        # re-norm
        one_hot_pseudo = F.one_hot(pseudo_label.to(int), num_classes=10).detach().clone()
        one_hot_comp_y = F.one_hot(comp_y.to(int)).detach().clone()
        revisedY0 = (1 - one_hot_comp_y).clone()
        revisedY0 = revisedY0 * pred_t_w
        revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(10,1).transpose(0,1)
        
        loss_cl = scl_nl(logit_t_w, comp_y.long()) #+ scl_nl(logit_t_s, comp_y.long())
        # calculate cl_proto_loss
        pred_cl = F.softmax(logits_com, dim=1)
        loss_proto_cl = -(torch.log(1.-pred_cl)*one_hot_pseudo).sum(1).mean()
        # calculate kl_div
        soft_positive_label0 = revisedY0.detach().clone()
        # one_hot
        threshold = 0.6  
        max_vals, max_indices = torch.max(soft_positive_label0, dim=1)
        indices_above_threshold = max_vals > threshold
        one_hot_softlabel = torch.zeros_like(soft_positive_label0)
        one_hot_softlabel[indices_above_threshold, max_indices[indices_above_threshold]] = 1
        soft_positive_label0[indices_above_threshold] = one_hot_softlabel[indices_above_threshold].clone()
        soft_positive_label0 = soft_positive_label0.detach().clone()

        soft_positive_label1 = revisedY0.detach().clone()

        loss_kl = kl_div(pred_t_s.log(), soft_positive_label0)

        # calculate proto_loss
        loss_proto = kl_div(F.log_softmax(torch.div(logits_prot, args.tau_proto), dim=1), soft_positive_label0)
        # calculate loss
        lam = min((epoch/100)*args.lam, args.lam) * 0.1
        loss = loss_cl + lam*(loss_kl + loss_proto) + lam*10*loss_proto_cl
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        # logging
        losses.update(loss.item(), image_w.size(0))
        losses_cl.update(loss_cl.item(), image_w.size(0))
        losses_kl.update(loss_kl.item(), image_w.size(0))
        losses_proto.update(loss_proto.item(), image_w.size(0))

        if i % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLoss_cl:{:.6f}\tLoss_kl:{:.6f}\tLoss_proto:{:.6f}'.format(
                epoch, i * len(image_w), len(train_loader.dataset), 100. * i / len(train_loader), 
                losses.avg,
                losses_cl.avg,
                losses_kl.avg,
                losses_proto.avg
                ))
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Loss/train_kl', loss_kl.item(), epoch)
    writer.add_scalar('Loss/train_cl', loss_cl.item(), epoch)
    writer.add_scalar('Loss/train_proto', loss_proto.item(), epoch)
    writer.add_scalar('Loss/train_cl_proto', loss_proto_cl.item(), epoch)

    model.save_prototypes(epoch)
    return losses.avg

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,_ = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def clkd_scheduler(epoch, alpha):
    if epoch<=50:
        alpha = 1
    elif epoch<=150 and epoch>50:
        alpha = 1 - (epoch-50)/100
    else:
        alpha = 0
    return alpha

best_acc = 0
Acc = []
for epoch in range(0, args.epochs):
    logging.info('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    trainloss = train(train_loader, model, optimizer, epoch)
    scheduler.step()
    # eval
    model.eval()
    _, acc = eval_test(model.model, device, test_loader)
    if best_acc<acc:
        best_acc = acc
        torch.save(model.state_dict(), 'cifar10_best.pth')
    Acc.append(acc)
    print("Best Accuracy:{}".format(best_acc))

np.savetxt('acc.txt', np.array(Acc))

writer.close()
