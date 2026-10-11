from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.autograd import Variable
import os, errno
import numpy as np
from scipy import linalg


import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from itertools import repeat, cycle
import data

import os
import random
import numpy as np
import torch

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Utility functions and classes"""

import sys
import numpy as np
import torch

def apply_zca(data, zca_mean, zca_components):
        temp = data.numpy()
        #temp = temp.transpose(0,2,3,1)
        shape = temp.shape
        temp = temp.reshape(-1, shape[1]*shape[2]*shape[3])
        temp = np.dot(temp - zca_mean, zca_components.T)
        temp = temp.reshape(-1, shape[1], shape [2], shape[3])
        #temp = temp.transpose(0, 3, 1, 2)
        data = torch.from_numpy(temp).float()
        return data
        #print (temp)


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def assert_exactly_one(lst):
    assert sum(int(bool(el)) for el in lst) == 1, ", ".join(str(el)
                                                            for el in lst)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}


class AverageMeter:
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

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


def export(fn):
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


def parameter_count(module):
    return sum(int(param.numel()) for param in module.parameters())

def to_var(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad=requires_grad)

def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)


def make_dir_if_not_exists(path):
    """Make directory if doesn't already exists"""
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def denorm(x):
    out = (x+1)/2
    return out.clamp(0,1)
import sys
import numpy as np
import torch
def apply_zca(data, zca_mean, zca_components):
        temp = data.numpy()
        shape = temp.shape
        temp = temp.reshape(-1, shape[1]*shape[2]*shape[3])
        temp = np.dot(temp - zca_mean, zca_components.T)
        temp = temp.reshape(-1, shape[1], shape [2], shape[3])
        data = torch.from_numpy(temp).float()
        return data
        #print (temp)


def to_one_hot(inp):
    y_onehot = torch.FloatTensor(inp.size(0), 10)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    return Variable(y_onehot.cuda(),requires_grad=False)

"""
def mixup_data(input, target, lam):
    
    lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
    lam = Variable(lam)
    indices = np.random.permutation(input.size(0))
    input = input*lam.expand_as(input) + input[indices]*(1-lam.expand_as(input))
    target = target* lam.expand_as(target) + target*(1 - lam.expand_as(target))
    return input, target
"""

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_labelled_unlabelled(input_l, input_u, target_l, target_u, mixup_alpha):
    
    if mixup_alpha > 0.:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
    else:
        lam = 1.
    
    lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
    lam = Variable(lam)
    #lam = torch.max(lam, 1-lam)
    #indices = np.random.permutation(out.size(0))
    out = input_l*lam.expand_as(input_l) + input_u*(1-lam.expand_as(input_u))
    target_l = to_one_hot(target_l)
    target = target_l* lam.expand_as(target_l) + target_u*(1 - lam.expand_as(target_u))
    return out, target


def mixup_data_hidden(input, target,  mixup_alpha):
    if mixup_alpha > 0.:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
    else:
        lam = 1.
    lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
    lam = Variable(lam)
    indices = np.random.permutation(input.size(0))
    #target = to_one_hot(target)
    output = input*lam.expand_as(input) + input[indices]*(1-lam.expand_as(input))
    target_a, target_b = target ,target[indices]
    
    return output, target_a, target_b, lam



def load_data_subset(data_aug, batch_size,workers,dataset, data_target_dir, labels_per_class=100, valid_labels_per_class = 500):
    
    
    ## copied from GibbsNet_pytorch/load.py
    import numpy as np
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
        
    if dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    elif dataset == 'svhn':
        mean = [x / 255 for x in [127.5, 127.5, 127.5]]
        std = [x / 255 for x in [127.5, 127.5, 127.5]]
    elif dataset == 'mnist' or dataset == 'fashion':
        pass 
    elif dataset == 'cinic':
        pass 
    elif dataset == 'covid2':
        pass     
    elif dataset == 'covid3':
        pass         
    elif dataset == 'brain4':
        pass         
    elif dataset == 'tiny':
        pass     
    elif dataset == 'stl10':
        pass         
    elif dataset == 'imagenet':
        pass         
    else:
        assert False, "Unknow dataset : {}".format(dataset)
    
    if data_aug==1:
        print ('data aug')
        if dataset == 'svhn':
            train_transform = transforms.Compose(
                                             [ transforms.RandomCrop(32, padding=2), transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                            [transforms.ToTensor(), transforms.Normalize(mean, std)])
        elif dataset == 'mnist' or dataset == 'fashion' :
            hw_size = 28
            train_transform = transforms.Compose([
                                transforms.RandomCrop(hw_size),                
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                           ])
            test_transform = transforms.Compose([
                                transforms.CenterCrop(hw_size),                       
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                           ])
        elif dataset == 'cifar10':        
            train_transform = transforms.Compose([  # # ن¸؛ن»€ن¹ˆه¼„ن¸ھtwice
                        data.RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

            test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])
        elif dataset == 'imagenet':        
            train_transform = transforms.Compose([  # # ن¸؛ن»€ن¹ˆه¼„ن¸ھtwice
                        data.RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

            test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])           
        elif dataset == 'cinic':        
            train_transform = transforms.Compose([  # # ن¸؛ن»€ن¹ˆه¼„ن¸ھtwice
                        data.RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

            test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])     
                        #transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])

                        #transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
        elif dataset == 'covid2':        
            train_transform = transforms.Compose([  # # ن¸؛ن»€ن¹ˆه¼„ن¸ھtwice
                        transforms.CenterCrop(64), 
                        data.RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

            test_transform = transforms.Compose([
                        transforms.CenterCrop(64),                 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])          
             
            data_transformcovid = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((64,64)),
                        #transforms.RandomAffine(degrees=(0,0), translate=(0.1, 0.1)),
                        #transforms.RandomCrop(64, padding=4),
                        #transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])

            data_transformcovidun = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((64,64)),
                        #transforms.RandomAffine(degrees=(0,0), translate=(0.1, 0.1)),
                        #transforms.RandomCrop(64, padding=4),
                        #transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])


        elif dataset == 'covid3':        
            train_transform = transforms.Compose([  # # ن¸؛ن»€ن¹ˆه¼„ن¸ھtwice
                        transforms.CenterCrop(64), 
                        data.RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

            test_transform = transforms.Compose([
                        transforms.CenterCrop(64),                 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])          
             
            data_transformcovid = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((32,32)),
                        #transforms.RandomAffine(degrees=(0,0), translate=(0.1, 0.1)),
                        #transforms.RandomCrop(32, padding=4),
                        #transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])

            data_transformcovidun = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((64,64)),
                        #transforms.RandomAffine(degrees=(0,0), translate=(0.1, 0.1)),
                        #transforms.RandomCrop(64, padding=4),
                        #transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])

        elif dataset == 'brain4':        
            train_transform = transforms.Compose([  # # ن¸؛ن»€ن¹ˆه¼„ن¸ھtwice
                        transforms.CenterCrop(32), 
                        data.RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

            test_transform = transforms.Compose([
                        transforms.CenterCrop(32),                 
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])          
             
            data_transformcovid = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((32,32)),
                        #transforms.RandomAffine(degrees=(0,0), translate=(0.1, 0.1)),
                        #transforms.RandomCrop(32, padding=4),
                        #transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])

            data_transformcovidun = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                        transforms.Resize((32,32)),
                        #transforms.RandomAffine(degrees=(0,0), translate=(0.1, 0.1)),
                        #transforms.RandomCrop(64, padding=4),
                        #transforms.RandomRotation(10),
                        transforms.ToTensor(),
                        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Normalize(mean=(0.5,), std=(0.5,))
                    ])
        elif dataset == 'tiny':        
            train_transform = transforms.Compose([  # # ن¸؛ن»€ن¹ˆه¼„ن¸ھtwice
                        data.RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

            test_transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])              
        elif dataset == 'stl10':        
            train_transform = transforms.Compose([  # # ن¸؛ن»€ن¹ˆه¼„ن¸ھtwice
                        transforms.CenterCrop(96),  
                        data.RandomTranslateWithReflect(4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])

            test_transform = transforms.Compose([
                        transforms.CenterCrop(96),  
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                    ])   

            
                  
        else:    
            train_transform = transforms.Compose(
                                                 [transforms.RandomHorizontalFlip(),
                                                  transforms.RandomCrop(32, padding=2),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    else:
        print ('no data aug')
        if dataset == 'mnist' or dataset == 'fashion':
            hw_size = 28
            train_transform = transforms.Compose([
                                transforms.ToTensor(),       
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
            test_transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                           ])
                
        else:   
            train_transform = transforms.Compose(
                                                 [transforms.ToTensor(),
                                                 transforms.Normalize(mean, std)])
            test_transform = transforms.Compose(
                                                [transforms.ToTensor(), transforms.Normalize(mean, std)])
    if dataset == 'cifar10':
        train_data = datasets.CIFAR10(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR10(data_target_dir, train=False, transform=test_transform, download=True)
        #train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/cifar10/train+val', train_transform)
        #test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/cifar10/test', test_transform)


        num_classes = 10
    elif dataset == 'cinic':
        train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/CINIC10/train', train_transform)
        test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/CINIC10/test', test_transform)
        #train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/CINIC10/cinicorginal/train', train_transform)        
        #test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/CINIC10/cinicorginal/test', test_transform)
        
        num_classes = 10  
    elif dataset == 'covid2':
        #train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/train', train_transform)
        #test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/test', test_transform)
        train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/train', data_transformcovid)
        test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/test', data_transformcovid)
        num_classes = 2  
    elif dataset == 'covid3':
        #train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/train', train_transform)
        #test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/test', test_transform)
        train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid3/train', data_transformcovid)
        test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid3/test', data_transformcovid)
        num_classes = 3     
    elif dataset == 'brain4':
        #train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/train', train_transform)
        #test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/test', test_transform)
        train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/braino7000/train', data_transformcovid)
        test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/braino7000/test', data_transformcovid)
        num_classes = 4                              
    elif dataset == 'tiny':
        train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/tiny-imagenet-200/tiny-imagenet-200/train', train_transform)
        test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/tiny-imagenet-200/tiny-imagenet-200/test', test_transform)
        #train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/CINIC10/cinicorginal/train', train_transform)        
        #test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/CINIC10/cinicorginal/test', test_transform)
        
        num_classes = 200    
    elif dataset == 'stl10':
        train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/stl10/train', train_transform)
        test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/stl10/test', test_transform)
        #train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/CINIC10/cinicorginal/train', train_transform)        
        #test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/CINIC10/cinicorginal/test', test_transform)
        
        num_classes = 10            
    elif dataset == 'imagenet':
        train_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/imagenetred/train', train_transform)
        test_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/data/imagenetred/test', test_transform)
        num_classes = 1000   

    elif dataset == 'cifar100':
        train_data = datasets.CIFAR100(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.CIFAR100(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 100
    elif dataset == 'svhn':
        train_data = datasets.SVHN(data_target_dir, split='train', transform=train_transform, download=True)
        test_data = datasets.SVHN(data_target_dir, split='test', transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'mnist':
        train_data = datasets.MNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.MNIST(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif dataset == 'fashion':
        train_data = datasets.FashionMNIST(data_target_dir, train=True, transform=train_transform, download=True)
        test_data = datasets.FashionMNIST(data_target_dir, train=False, transform=test_transform, download=True)
        num_classes = 10        
    #print ('svhn', train_data.labels.shape)
    elif dataset == 'imagenet':
        assert False, 'Do not finish imagenet code'
    else:
        assert False, 'Do not support dataset : {}'.format(dataset)

        
    n_labels = num_classes
    
    def get_sampler(labels, n=None, n_valid= None):
        set_all_seeds(10)

        # Only choose digits in n_labels
        # n = number of labels per class for training
        # n_val = number of lables per class for validation
        #print type(labels)
        #print (n_valid)
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))
        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        
        indices_valid = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n_valid] for i in range(n_labels)])
        indices_train = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:n_valid+n] for i in range(n_labels)])
        indices_unlabelled = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[n_valid:] for i in range(n_labels)])
        #print (indices_train.shape)
        #print (indices_valid.shape)
        #print (indices_unlabelled.shape)
        indices_train = torch.from_numpy(indices_train)
        indices_valid = torch.from_numpy(indices_valid)
        indices_unlabelled = torch.from_numpy(indices_unlabelled)
        sampler_train = SubsetRandomSampler(indices_train)
        sampler_valid = SubsetRandomSampler(indices_valid)
        sampler_unlabelled = SubsetRandomSampler(indices_unlabelled)
        return sampler_train, sampler_valid, sampler_unlabelled
    
    #print type(train_data.train_labels)
    
    # Dataloaders for MNIST
    if dataset == 'svhn':
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.labels, labels_per_class, valid_labels_per_class)
    elif dataset == 'mnist' or dataset == 'fashion' :
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.train_labels.numpy(), labels_per_class, valid_labels_per_class)
    else: 
        train_sampler, valid_sampler, unlabelled_sampler = get_sampler(train_data.targets, labels_per_class, valid_labels_per_class)

    labelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = train_sampler,  num_workers=workers, pin_memory=True)
    validation = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = valid_sampler,  num_workers=workers, pin_memory=True)
    unlabelled = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler = unlabelled_sampler,  num_workers=workers, pin_memory=True)
    test = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    if dataset == 'stl10':
        unlabeled_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/stl10/unlabeled', test_transform)
        unlabelled = torch.utils.data.DataLoader(unlabeled_data, batch_size=batch_size, num_workers=workers, pin_memory=True)
    #if dataset == 'covid2':

        #unlabeled_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid2/train', data_transformcovidun)
        #unlabelled = torch.utils.data.DataLoader(unlabeled_data, batch_size=batch_size, num_workers=workers, pin_memory=True)
    #if dataset == 'covid3':

        #unlabeled_data = torchvision.datasets.ImageFolder('E:/3070/boosting/gan/proposal-iraji/paper01-vc/ICT-master/ICT-master/data/covid3/train', data_transformcovidun)
        #unlabelled = torch.utils.data.DataLoader(unlabeled_data, batch_size=batch_size, num_workers=workers, pin_memory=True)        
    return labelled, validation, unlabelled, test, num_classes

if __name__ == '__main__':
    labelled, validation, unlabelled, test, num_classes  = load_data_subset(data_aug=1, batch_size=32,workers=1,dataset='cifar10', data_target_dir="/u/vermavik/data/DARC/cifar10", labels_per_class=100, valid_labels_per_class = 500)
    for (inputs, targets), (u, _) in zip(cycle(labelled), unlabelled):
        print (input)














