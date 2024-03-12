import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from randaugment import RandomAugment
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # device

def generate_compl_labels(labels):
    # args, labels: ordinary labels
    labels_Y = labels.to(device)
    a=torch.nonzero(labels==1)
    labels=a[:,1]
    K = torch.max(labels)+1
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, int(K.item())), len(labels), 0)
    mask = np.ones((len(labels), int(K.item())), dtype=bool)  # mask: (len(labels), K)
    mask[range(len(labels)), labels.cpu().numpy()] = False
    candidates_ = candidates[mask].reshape(len(labels), K-1)  # this is the candidates without true class
    idx = np.random.randint(0, K-1, len(labels))
    complementary_labels = candidates_[np.arange(len(labels)), np.array(idx)]
    return torch.from_numpy(complementary_labels).cpu()

def load_cifar10(batch_size):   # load data

    test_transform = transforms.Compose(
            [transforms.ToTensor(),  # change to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])   # normalize (no need change)
    
    temp_train = dsets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())  # train
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long() # target to tensor
    # get original data and labels

    test_dataset = dsets.CIFAR10(root='./data', train=False, transform=test_transform)  # test
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4)
    # set test dataloader
    one_hot_labels = torch.nn.functional.one_hot(labels)
    comp = generate_compl_labels(one_hot_labels)
    partial_matrix_dataset = CIFAR10_Augmentention(data, comp.float(), labels.float())
    # generate partial label dataset include data, comp label, label
    
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset,  # load train data
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    return partial_matrix_train_loader ,test_loader

def load_svhn(batch_size):   # load data

    test_transform = transforms.Compose(
            [transforms.ToTensor(),  # change to tensor
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])   # normalize (no need change)
    
    temp_train = dsets.SVHN(root='./data/SVHN', split='train', download=True, transform=transforms.ToTensor())
    test_dataset = dsets.SVHN(root='./data/SVHN', split='test', download=True, transform=transforms.ToTensor())
    temp_train.targets = temp_train.labels
    temp_train.classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four', '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    data, labels = temp_train.data, torch.Tensor(temp_train.targets).long() # target to tensor
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*4, shuffle=False, num_workers=4)

    one_hot_labels = torch.nn.functional.one_hot(labels)
    comp = generate_compl_labels(one_hot_labels)
    data = data.reshape(len(data), 32, 32, 3)
    partial_matrix_dataset = SVHN_Augmentention(data, comp.float(), labels.float())
    
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset,  # load train data
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True)
    return partial_matrix_train_loader ,test_loader


class CIFAR10_Augmentention(Dataset):   # augmentation
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index
    
class SVHN_Augmentention(Dataset):   # augmentation
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels
        self.weak_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        self.strong_transform = transforms.Compose(
            [
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            RandomAugment(3, 5),
            transforms.ToTensor(), 
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.weak_transform(self.images[index])
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_w, each_image_s, each_label, each_true_label, index
