from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
import os
os.makedirs("visualizations/confusion", exist_ok=True)
os.makedirs("visualizations/tsne", exist_ok=True)

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
    confusion_matrix,
    balanced_accuracy_score
)

import pandas as pd
os.makedirs("metrics", exist_ok=True)


# =========================
# 2. IMPORTS
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import os
os.makedirs("diffusion_outputs", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =========================
# 3. DATA (Scalable)
# =========================
image_size = 32   # scalable



# =========================
# 4. VPSDE
# =========================
class VPSDE:
    def __init__(self, beta_min=0.1, beta_max=10, N=1000):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_prob(self, x, t):
        log_mean = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        mean = torch.exp(log_mean[:, None, None, None]) * x
        std = torch.sqrt(1 - torch.exp(2 * log_mean))
        return mean, std

    def diffusion_coeff(self, t):
        return torch.sqrt(self.beta(t))

sde = VPSDE(N=2000)


# =========================
# 5. TIME EMBEDDING
# =========================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

# =========================
# 6. RESIDUAL BLOCK
# =========================
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.time_mlp = nn.Linear(time_dim, out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h += self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.shortcut(x)

# =========================
# 7. PRODUCTION UNET
# =========================
class UNet(nn.Module):
    def __init__(self, img_ch=3, base_ch=128, time_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU()
        )

        self.conv0 = nn.Conv2d(img_ch, base_ch, 3, padding=1)

        self.down1 = ResBlock(base_ch, base_ch*2, time_dim)
        self.down2 = ResBlock(base_ch*2, base_ch*4, time_dim)

        self.pool = nn.MaxPool2d(2)

        self.mid = ResBlock(base_ch*4, base_ch*4, time_dim)

        self.up1 = ResBlock(base_ch*4 + base_ch*4, base_ch*2, time_dim)
        self.up2 = ResBlock(base_ch*2 + base_ch*2, base_ch, time_dim)

        self.out = nn.Conv2d(base_ch, img_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        x0 = self.conv0(x)
        x1 = self.down1(x0, t_emb)
        x2 = self.pool(x1)

        x3 = self.down2(x2, t_emb)
        x4 = self.pool(x3)

        mid = self.mid(x4, t_emb)

        up1 = F.interpolate(mid, scale_factor=2)
        up1 = torch.cat([up1, x3], dim=1)
        up1 = self.up1(up1, t_emb)

        up2 = F.interpolate(up1, scale_factor=2)
        up2 = torch.cat([up2, x1], dim=1)
        up2 = self.up2(up2, t_emb)

        return self.out(up2)

modeldif = UNet(img_ch=3).to(device)

# =========================
# 8. EMA
# =========================
ema_modeldif = copy.deepcopy(modeldif)
ema_decay = 0.999

def update_ema(model, ema_model):
    with torch.no_grad():
        for p, ema_p in zip(model.parameters(), ema_model.parameters()):
            ema_p.data = ema_decay * ema_p.data + (1 - ema_decay) * p.data

optimizerdif = torch.optim.Adam(modeldif.parameters(), lr=2e-4)

def update_classifier_ema(model, ema_model, decay=0.999):

    with torch.no_grad():

        msd = model.state_dict()
        emsd = ema_model.state_dict()

        for k in emsd.keys():

            if emsd[k].dtype.is_floating_point:

                emsd[k].mul_(decay).add_(
                    msd[k],
                    alpha=1 - decay
                )
            else:
                emsd[k] = msd[k]
# =========================
# 9. LOSS (Score Matching)
# =========================
def loss_fn(model, x):
    batch_size = x.shape[0]
    t = torch.rand(batch_size, device=device) * 0.999 + 0.001
    z = torch.randn_like(x)

    mean, std = sde.marginal_prob(x, t)
    perturbed = mean + std[:, None, None, None] * z

    score = model(perturbed, t)
    loss = torch.mean((score * std[:, None, None, None] + z) ** 2)
    return loss



import re
import argparse
import os
import shutil
import time
import math
import logging
import torchvision.utils as vutils

import os, gzip, torch
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from vat import VATLoss

def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


from utils import *
import losses
import ramps

NO_LABEL=-1


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys, os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import mixup_data


## code for CNN13 from https://github.com/benathi/fastswa-semi-sup/blob/master/mean_teacher/architectures.py
from torch.nn.utils import weight_norm

class CNN13(nn.Module):
    """CNN13 with a MarginGAN-compatible 128-D feature extractor."""
    def __init__(self, num_classes=10, dropout=0.5):
        super(CNN13, self).__init__()
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3,128,3,padding=1)); self.bn1a=nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128,128,3,padding=1)); self.bn1b=nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128,128,3,padding=1)); self.bn1c=nn.BatchNorm2d(128)
        self.mp1=nn.MaxPool2d(2,2); self.drop1=nn.Dropout(dropout)
        self.conv2a=weight_norm(nn.Conv2d(128,256,3,padding=1)); self.bn2a=nn.BatchNorm2d(256)
        self.conv2b=weight_norm(nn.Conv2d(256,256,3,padding=1)); self.bn2b=nn.BatchNorm2d(256)
        self.conv2c=weight_norm(nn.Conv2d(256,256,3,padding=1)); self.bn2c=nn.BatchNorm2d(256)
        self.mp2=nn.MaxPool2d(2,2); self.drop2=nn.Dropout(dropout)
        self.conv3a=weight_norm(nn.Conv2d(256,512,3,padding=0)); self.bn3a=nn.BatchNorm2d(512)
        self.conv3b=weight_norm(nn.Conv2d(512,256,1)); self.bn3b=nn.BatchNorm2d(256)
        self.conv3c=weight_norm(nn.Conv2d(256,128,1)); self.bn3c=nn.BatchNorm2d(128)
        self.ap3=nn.AvgPool2d(6,stride=2,padding=0)
        self.fc1=weight_norm(nn.Linear(128,num_classes))

    def extract_features(self,x):
        x=self.conv1a(x); x=self.bn1a(x); x=self.activation(x)
        x=self.conv1b(x); x=self.bn1b(x); x=self.activation(x)
        x=self.conv1c(x); x=self.bn1c(x); x=self.activation(x)
        x=self.mp1(x); x=self.drop1(x)
        x=self.conv2a(x); x=self.bn2a(x); x=self.activation(x)
        x=self.conv2b(x); x=self.bn2b(x); x=self.activation(x)
        x=self.conv2c(x); x=self.bn2c(x); x=self.activation(x)
        x=self.mp2(x); x=self.drop2(x)
        x=self.conv3a(x); x=self.bn3a(x); x=self.activation(x)
        x=self.conv3b(x); x=self.bn3b(x); x=self.activation(x)
        x=self.conv3c(x); x=self.bn3c(x); x=self.activation(x)
        x=self.ap3(x)
        x=x.view(x.size(0),-1)
        return x

    def forward(self,x,target=None,mixup_hidden=False,mixup_alpha=0.1,layers_mix=None):
        if mixup_hidden:
            layer_mix=random.randint(0,layers_mix); out=x
            if layer_mix==0: out,y_a,y_b,lam=mixup_data(out,target,mixup_alpha)
            out=self.activation(self.bn1a(self.conv1a(out))); out=self.activation(self.bn1b(self.conv1b(out))); out=self.bn1c(self.conv1c(out))
            if layer_mix==1: out,y_a,y_b,lam=mixup_data(out,target,mixup_alpha)
            out=self.activation(out); out=self.drop1(self.mp1(out))
            out=self.activation(self.bn2a(self.conv2a(out))); out=self.activation(self.bn2b(self.conv2b(out))); out=self.bn2c(self.conv2c(out))
            if layer_mix==2: out,y_a,y_b,lam=mixup_data(out,target,mixup_alpha)
            out=self.activation(out); out=self.drop2(self.mp2(out))
            out=self.activation(self.bn3a(self.conv3a(out))); out=self.activation(self.bn3b(self.conv3b(out))); out=self.bn3c(self.conv3c(out))
            if layer_mix==3: out,y_a,y_b,lam=mixup_data(out,target,mixup_alpha)
            out=self.activation(out); out=self.ap3(out); out=out.view(out.size(0),-1); out=self.fc1(out)
            lam=torch.tensor(lam,device=out.device).repeat(y_a.size())
            return out,y_a,y_b,lam
        feat=self.extract_features(x)
        return self.fc1(feat)


def cnn13(num_classes=10, dropout = 0.0):
    model = CNN13(num_classes = num_classes, dropout=dropout)
    return model


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
set_all_seeds(10)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')
def save_images(images, size, image_path):
    return imsave(images, size, image_path)



def imsave(images, size, path): # # #
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, (255.0 * image).astype('uint8'))
def generated_weight(epoch):
    alpha = 0.0
    T1 = 10
    T2 = 60
    af = 0.3
    if epoch > T1:
        alpha = (epoch-T1) / (T2-T1)*af
        if epoch > T2:
            alpha = af
    return alpha

def nll_loss_neg(y_pred, y_true,CLASS_WEIGHT):
    out = torch.sum((y_true * y_pred), dim=1)
    return torch.mean(- torch.log((1 - out) + 1e-6))

def nll_loss_neg2(y_pred, y_true):
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log(( out) + 1e-6))

def nll_loss_neg3(y_pred, y_true):
    out = torch.sum(y_true * y_pred, dim=1)
    return torch.mean(- torch.log((1 - out) + 1e-6))

wcw2 = np.ones(10)
print(wcw2)
wcw2 = torch.Tensor(wcw2 ).cuda()

transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                transforms.RandomAffine(0, (1/8,0))]) # max horizontal shift by 4


mixup=0.48

def mixup_batch(mixup,real1,real2,fake):
                def one_batch():


                    data = torch.cat((real1,real2, fake))
                    ones1 = Variable(torch.ones(real1.size(0), 1))
                    ones2 = Variable(torch.ones(real2.size(0), 1))                    
                    zeros = Variable(torch.zeros(fake.size(0), 1))

                    perm = torch.randperm(data.size(0)).view(-1).long()
                    if True:
                        ones1 = ones1.cuda()
                        ones2 = ones2.cuda()
                        zeros = zeros.cuda()
                        perm = perm.cuda()
                    labels = torch.cat((ones1,ones2, zeros))
                    return data[perm], labels[perm]

                d1, l1 = one_batch()
                if mixup == 0:
                    return d1, l1
                d2, l2 = one_batch()
                alpha = Variable(torch.tensor(np.random.beta(mixup, mixup)) )

                #print(alpha)

                if True:
                    alpha = alpha.cuda()
                d = alpha * d1 + (1. - alpha) * d2
                l = alpha * l1 + (1. - alpha) * l2
                return d, l

def mixup_batch2(mixup,f,y):
                def one_batch():


                    data = f
                    perm = torch.randperm(data.size(0)).view(-1).long()
                    if True:
                        perm = perm.cuda()
                    labels =y 
                    return data[perm], labels[perm]

                d1, l1 = one_batch()
                if mixup == 0:
                    return d1, l1
                d2, l2 = one_batch()
                alpha = Variable(torch.tensor(np.random.beta(mixup, mixup)) )

                #print(alpha)

                if True:
                    alpha = alpha.cuda()
                d = alpha * d1 + (1. - alpha) * d2
                l = alpha * l1 + (1. - alpha) * l2
                return d, l


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
# batch_size * input_dim => batch_size * output_dim * input_size * input_size
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)

        return x



# batch_size * input_dim * input_size * input_size => batch_size * output_dim
class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 500),
        
        )
        initialize_weights(self)
        self.ln=nn.Linear(500, self.output_dim)
        self.sig=nn.Sigmoid()
        self.features=None

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        self.features=x.view(-1, x.size(1))
        output=self.ln(self.features)
        output=self.sig(output)
        return output












import re
import argparse
import os
import shutil
import time
import math
from itertools import repeat, cycle
import matplotlib as mpl
mpl.use('Agg')

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from collections import OrderedDict
import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import _pickle as pickle


from utils import *
#from networks.wide_resnet import *
#from networks.wide_resnet13 import *
#from networks.lenet import *
#from networks.lenet14 import *

#from cmn import *



import losses


parser = argparse.ArgumentParser(description='Interpolation consistency training')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                        choices=['cifar10','svhn','mnist','cinic','tiny','stl10','covid2','covid3','brain4','fashion'],
                        help='dataset: cifar10 or svhn' )
parser.add_argument('--num_labeled', default=1000, type=int, metavar='L',
                    help='number of labeled samples per class')
parser.add_argument('--num_valid_samples', default=1000, type=int, metavar='V',
                    help='number of validation samples per class')
parser.add_argument('--arch', default='cnn13', type=str, help='either of cnn13,c, WRN28_2 , cifar_shakeshake26')
parser.add_argument('--dropout', default=0.0, type=float,
                    metavar='DO', help='dropout rate')

parser.add_argument('--sl', action='store_true',
                    help='only supervised learning: no use of unlabeled data')
parser.add_argument('--pseudo_label', choices=['single','mean_teacher'],
                        help='pseudo label generated from either a single model or mean teacher model')
parser.add_argument('--optimizer', type = str, default = 'sgd',
                        help='optimizer we are going to use. can be either adam of sgd')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='max learning rate')
parser.add_argument('--initial_lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate when using linear rampup')
parser.add_argument('--lr_rampup', default=0, type=int, metavar='EPOCHS',
                    help='length of learning rate rampup in the beginning')
parser.add_argument('--lr_rampdown_epochs', default=None, type=int, metavar='EPOCHS',
                    help='length of learning rate cosine rampdown (>= length of training): the epoch at which learning rate \
                    reaches to zero')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', action='store_true',
                    help='use nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--ema_decay', default=0.999, type=float, metavar='ALPHA',
                    help='ema variable decay rate (default: 0.999)')
parser.add_argument('--mixup_consistency', default=1.0, type=float,
                    help='consistency coeff for mixup usup loss')
parser.add_argument('--consistency_type', default="mse", type=str, metavar='TYPE',
                    choices=['mse', 'kl'],
                    help='consistency loss type to use')
parser.add_argument('--consistency_rampup_starts', default=30, type=int, metavar='EPOCHS',
                    help='epoch at which consistency loss ramp-up starts')
parser.add_argument('--consistency_rampup_ends', default=30, type=int, metavar='EPOCHS',
                    help='lepoch at which consistency loss ramp-up ends')
parser.add_argument('--mixup_sup_alpha', default=0.0, type=float,
                    help='for supervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
parser.add_argument('--mixup_usup_alpha', default=0.0, type=float,
                    help='for unsupervised loss, the alpha parameter for the beta distribution from where the mixing lambda is drawn')
parser.add_argument('--mixup_hidden', action='store_true',
                    help='apply mixup in hidden layers')
parser.add_argument('--num_mix_layer', default=3, type=int,
                    help='number of layers on which mixup is applied including input layer')
parser.add_argument('--checkpoint_epochs', default=50, type=int,
                    metavar='EPOCHS', help='checkpoint frequency in epochs, 0 to turn checkpointing off (default: 1)')
parser.add_argument('--evaluation_epochs', default=1, type=int,
                    metavar='EPOCHS', help='evaluation frequency in epochs, 0 to turn evaluation off (default: 1)')
parser.add_argument('--print_freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate model on evaluation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--root_dir', type = str, default = 'experiments',
                        help='folder where results are to be stored')
parser.add_argument('--data_dir', type = str, default = 'data/cifar10/',
                        help='folder where data is stored')
parser.add_argument('--n_cpus', default=0, type=int,
                    help='number of cpus for data loading')
parser.add_argument('--job_id', type=str, default='')
parser.add_argument('--add_name', type=str, default='')


args = parser.parse_args()
print (args)
use_cuda = torch.cuda.is_available()


best_prec1 = 0
global_step = 0
best_acc=0
##get number of updates etc#####
   
if args.dataset == 'mnist' or args.dataset == 'fashion':
        z_dim=62
        G = generator(input_dim=z_dim, output_dim=1, input_size=28)
        D = discriminator(input_dim=1, output_dim=1, input_size=28)   
             
else:    

        z_dim=100

        G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        D = discriminator(input_dim=3, output_dim=1, input_size=32)
if args.dataset == 'tiny':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=3, input_size=64)
        D = discriminator(input_dim=3, output_dim=1, input_size=64)
if args.dataset == 'stl10':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=3, input_size=96)
        D = discriminator(input_dim=3, output_dim=1, input_size=96)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32)            

if args.dataset == 'covid2':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=1, input_size=64)
        D = discriminator(input_dim=1, output_dim=1, input_size=64)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32) 

if args.dataset == 'covid3':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=1, input_size=32)
        D = discriminator(input_dim=1, output_dim=1, input_size=32)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32) 

if args.dataset == 'brain4':        
        z_dim=100

        G = generator(input_dim=z_dim, output_dim=1, input_size=32)
        D = discriminator(input_dim=1, output_dim=1, input_size=32)            
        #G = generator(input_dim=z_dim, output_dim=3, input_size=32)
        #D = discriminator(input_dim=3, output_dim=1, input_size=32)         
if args.resume=='y':
        D.load_state_dict(torch.load('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/experiments/s/_D.pkl'))
        G.load_state_dict(torch.load('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/experiments/s/_G.pkl'))

#else:    
        from dcgan import Discriminator, Generator

        #D   = Discriminator(1)
        #G = Generator(1)
        #z_dim=100
G.cuda()
D.cuda()

lrG=0.0002
lrD=0.0002
beta1=0.5
beta2=0.999
G_optimizer = torch.optim.Adam(G.parameters(), lr=lrG, betas=(beta1, beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=lrD, betas=(beta1, beta2))

BCEloss = nn.BCELoss().cuda()
l2loss = nn.MSELoss()



D.train()
G.train()
if args.dataset == 'cifar10':
    len_data = args.num_labeled
    #from dataloadercfar10 import dataloader
    num_updates = int((50000/args.batch_size))*args.epochs 
elif args.dataset == 'svhn':
    #from dataloadersvh import dataloader
    len_data = args.num_labeled
    num_updates = int((73250/args.batch_size)+1)*args.epochs
elif args.dataset == 'mnist':
    #from dataloadersvh import dataloader
    len_data = args.num_labeled
    num_updates = int((50000/args.batch_size)+1)*args.epochs     
    print ('number of updates', num_updates)
elif args.dataset == 'fashion':
    #from dataloadersvh import dataloader
    len_data = args.num_labeled
    num_updates = int((50000/args.batch_size)+1)*args.epochs     
    print ('number of updates', num_updates)
elif args.dataset == 'cinic':
    #from dataloadercinic import dataloader
    len_data = args.num_labeled
    num_updates = int((180000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)
elif args.dataset == 'covid2':
    #from dataloadercinic import dataloader
    len_data = args.num_labeled
    num_updates = int((5000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates) 
elif args.dataset == 'covid3':
    #from dataloadercinic import dataloader
    len_data = args.num_labeled
    num_updates = int((1000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)     
elif args.dataset == 'brain4':
    #from dataloadercinic import dataloader
    len_data = args.num_labeled
    num_updates = int((2700/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)               
elif args.dataset == 'tiny':
    #from dataloadercinic import dataloader
    len_data = args.num_labeled
    num_updates = int((100000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)    
elif args.dataset == 'stl10':
    #from dataloadercinic import dataloader
    len_data = args.num_labeled
    num_updates = int((105000/args.batch_size)+1)*args.epochs    
    #num_updates = int((90000/args.batch_size)+1)*args.epochs         
    print ('number of updates', num_updates)     
#print (args.batch_size, num_updates, args.epochs)

#### load data###
if args.dataset == 'cifar10':
    data_source_dir = args.data_dir
    #trainloader, unlabelledloader, testloader = dataloader('cifar10', 32, args.batch_size, args.num_labeled)
    #num_classes=10
    #validloader=testloader
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #zca_components = np.load(args.data_dir +'zca_components.npy')
    #zca_mean = np.load(args.data_dir +'zca_mean.npy')
if args.dataset == 'cinic':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cinic', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=10
    #validloader=testloader
    #trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #zca_components = np.load(args.data_dir +'zca_components.npy')
    #zca_mean = np.load(args.data_dir +'zca_mean.npy')
if args.dataset == 'covid2':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'covid2', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=2
if args.dataset == 'covid3':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'covid3', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=3    
    #validloader=testloader
    #trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #zca_components = np.load(args.data_dir +'zca_components.npy')
    #zca_mean = np.load(args.data_dir +'zca_mean.npy')    
if args.dataset == 'brain4':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'brain4', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=4        
if args.dataset == 'tiny':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'tiny', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=10
    #validloader=testloader
    #trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'cifar10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #zca_components = np.load(args.data_dir +'zca_components.npy')
    #zca_mean = np.load(args.data_dir +'zca_mean.npy')   
if args.dataset == 'stl10':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'stl10', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    num_classes=10         
if args.dataset == 'svhn':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'svhn', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
    #trainloader, unlabelledloader, testloader = dataloader('svhn', 32, args.batch_size, args.num_labeled)
    #num_classes=10
    #validloader=testloader
if args.dataset == 'mnist':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'mnist', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
if args.dataset == 'fashion':
    data_source_dir = args.data_dir
    trainloader, validloader, unlabelledloader, testloader, num_classes = load_data_subset(1, args.batch_size, args.n_cpus ,'fashion', data_source_dir, labels_per_class = args.num_labeled, valid_labels_per_class = args.num_valid_samples)
### lists for collecting output statistics###
### lists for collecting output statistics###
### lists for collecting output statistics###
train_class_loss_list = []
train_ema_class_loss_list = []
train_mixup_consistency_loss_list = []
train_mixup_consistency_coeff_list = []
train_error_list = []
train_ema_error_list = []
train_lr_list = []


val_class_loss_list = []
val_error_list = []
val_ema_class_loss_list = []
val_ema_error_list = []
diffusion_loss_list = []
metrics_rows = []


### get net####

# torch.randn(20, 3, 64, 64)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 512 * block.expansion
        self.conv1 = nn.Conv2d(1, 64, kernel_size=2, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        emb = out.view(out.size(0), -1)
        out = self.linear(emb)
        return out#, emb
    def get_embedding_dim(self):
        return self.embDim

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])


def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])






class classifier(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        initialize_weights(self)

        self.softmax = nn.Softmax()

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_classes),

        )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


#from cif import *
#from stl1chanal4 import *
def getNetwork(args, num_classes, ema= False):
    
    if args.arch in ['cnn13','WRN28_2']:
        net = eval(args.arch)(num_classes, args.dropout)
    elif args.arch in ['cmn']:
        net = cmn() 
    elif args.arch in ['c']:

        #net=cifar10(n_channel=128)        
        net = classifier(input_dim=1, output_dim=num_classes, input_size=28)         
    elif args.arch in ['res']:
        #net = ResNet18() 
        net=stl10(n_channel=32)                 
    elif args.arch in ['cifar_shakeshake26']:
        model_factory = architectures.__dict__[args.arch]
        model_params = dict(pretrained=args.pretrained, num_classes=num_classes)
        net = model_factory(**model_params)
    else:
        print('Error : Network should be either [cnn13/ WRN28_2 / cifar_shakeshake26')
        sys.exit(0)
    if args.resume=='y':
        net.load_state_dict(torch.load('E:/3070/boosting/gan/proposal-iraji/paper03/ICT-master/ICT-master/experiments/s/_C.pkl'))
    if ema:
        for param in net.parameters():
            param.detach_()

    return net



def experiment_name(sl = False,
                    dataset='cifar10',
                    labels  = 1000,
                    valid = 1000,
                    optimizer = 'sgd',
                    lr = 0.0001,
                    init_lr = 0.0,
                    lr_rampup = 5,
                    lr_rampdown = 10,
                    l2 = 0.0005,
                    ema_decay = 0.999,
                    mixup_consistency = 1.0,
                    consistency_type = 'mse',
                    consistency_rampup_s = 30,
                    consistency_rampup_e = 30,
                    mixup_sup_alpha = 1.0,
                    mixup_usup_alpha = 2.0,
                    mixup_hidden = False,
                    num_mix_layer = 3, 
                    pseudo_label = 'single',
                    epochs=10,
                    batch_size =100,
                    arch = 'WRN28_2',
                    dropout = 0.5, 
                    nesterov = True,
                    job_id=None,
                    add_name=''):
    if sl:
        exp_name = 'SL_'
    else:
        exp_name = 'SSL_'
    exp_name += str(dataset)
    exp_name += '_labels_' + str(labels)
    exp_name += '_valids_' + str(valid)
    
    exp_name += '_arch'+ str(arch)
    exp_name += '_do'+ str(dropout)
    exp_name += '_opt'+ str(optimizer)
    exp_name += '_lr_'+str(lr)
    exp_name += '_init_lr_'+ str(init_lr)
    exp_name += '_ramp_up_'+ str(lr_rampup)
    exp_name += '_ramp_dn_'+ str(lr_rampdown)
    
    exp_name += '_ema_d_'+ str(ema_decay)
    exp_name += '_m_consis_'+ str(mixup_consistency)
    exp_name += '_type_'+ str(consistency_type)
    exp_name += '_ramp_'+ str(consistency_rampup_s)
    exp_name += '_'+ str(consistency_rampup_e)
    
    
    exp_name += '_l2_'+str(l2)
    exp_name += '_eph_'+str(epochs)
    exp_name += '_bs_'+str(batch_size)
    
    if mixup_sup_alpha:
        exp_name += '_m_sup_a'+str(mixup_sup_alpha)
    if mixup_usup_alpha:
        exp_name += '_m_usup_a'+str(mixup_usup_alpha)
    if mixup_hidden :
        exp_name += 'm_hidden_'
        exp_name += str(num_mix_layer)
    exp_name += '_pl_'+str(pseudo_label)
    if nesterov:
        exp_name += '_nesterov_'
    if job_id!=None:
        exp_name += '_job_id_'+str(job_id)
    if add_name!='':
        exp_name += '_add_name_'+str(add_name)

    # exp_name += strftime("_%Y-%m-%d_%H:%M:%S", gmtime())
    print('experiement name: ' + exp_name)
    return exp_name

def mixup_data_sup(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    #x, y = x.numpy(), y.numpy()
    #mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_x = lam * x + (1 - lam) * x[index,:]
    #y_a, y_b = torch.Tensor(y).type(torch.LongTensor), torch.Tensor(y[index]).type(torch.LongTensor)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, mixed target, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = np.random.permutation(batch_size)
    x, y = x.data.cpu().numpy(), y.data.cpu().numpy()
    mixed_x = torch.Tensor(lam * x + (1 - lam) * x[index,:])
    mixed_y = torch.Tensor(lam * y + (1 - lam) * y[index,:])
    
    mixed_x = Variable(mixed_x.cuda())
    mixed_y = Variable(mixed_y.cuda())
    return mixed_x, mixed_y, lam





# ============================================================
# MarginGAN-style analysis/output helpers
# ============================================================
def _analysis_metrics(y_true, y_pred):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); cm=confusion_matrix(y_true,y_pred)
    sens=[]; spec=[]
    for i in range(cm.shape[0]):
        tp=cm[i,i]; fn=cm[i,:].sum()-tp; fp=cm[:,i].sum()-tp; tn=cm.sum()-(tp+fn+fp)
        sens.append(tp/(tp+fn+1e-8)); spec.append(tn/(tn+fp+1e-8))
    sensitivity=float(np.mean(sens)); specificity=float(np.mean(spec))
    return {'accuracy':float(accuracy_score(y_true,y_pred)),'sensitivity':sensitivity,'specificity':specificity,
            'balanced_accuracy':float(balanced_accuracy_score(y_true,y_pred)),
            'f1_score':float(f1_score(y_true,y_pred,average='macro',zero_division=0)),
            'gmean':float(np.sqrt(max(sensitivity*specificity,0.0)))}

def _collect_analysis_data(model,loader,max_samples=None):
    model.eval(); max_samples=len(loader.dataset) if max_samples is None else max_samples
    feats=[]; preds=[]; labels=[]; total=0
    with torch.no_grad():
        for x,y in loader:
            x=x.cuda(non_blocking=True) if use_cuda else x; y=y.cuda(non_blocking=True) if use_cuda else y
            feat=model.extract_features(x); logits=model(x); pred=torch.argmax(logits,dim=1)
            feats.append(feat.cpu().numpy()); preds.append(pred.cpu().numpy()); labels.append(y.cpu().numpy()); total+=len(x)
            if total>=max_samples: break
    return np.concatenate(feats),np.concatenate(preds),np.concatenate(labels)

def _collect_generated_analysis_data(model,x_generated,pseudo_labels):
    model.eval()
    with torch.no_grad():
        feat=model.extract_features(x_generated); pred=torch.argmax(model(x_generated),dim=1)
    return feat.cpu().numpy(),pred.cpu().numpy(),pseudo_labels.cpu().numpy()

def _save_analysis_confusion(y_true,y_pred,name,tag):
    cm=confusion_matrix(y_true,y_pred); plt.figure(figsize=(8,8)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',cbar=True)
    plt.title(f'{name} Confusion Matrix Epoch {tag}'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.savefig(os.path.join('visualizations/confusion',f'{name}_{tag}.pdf'),bbox_inches='tight'); plt.close()

def _save_analysis_tsne(features,labels,name,tag,max_samples=1500):
    if len(features)>max_samples:
        idx=np.linspace(0,len(features)-1,max_samples).astype(int); features=features[idx]; labels=labels[idx]
    if len(features)<3:return
    perplexity=min(30,max(2,len(features)-1)); features=StandardScaler().fit_transform(features)
    reduced=TSNE(n_components=2,perplexity=perplexity,random_state=42).fit_transform(features)
    plt.figure(figsize=(10,10)); sc=plt.scatter(reduced[:,0],reduced[:,1],c=labels,cmap='tab10',s=12); plt.colorbar(sc)
    plt.title(f'{name} t-SNE Epoch {tag}'); plt.savefig(os.path.join('visualizations/tsne',f'{name}_{tag}.pdf'),bbox_inches='tight'); plt.close()

def _append_training_metric(epoch,dataset_name,y_true,y_pred):
    m=_analysis_metrics(y_true,y_pred); m['epoch']=epoch; m['dataset']=dataset_name; metrics_rows.append(m)
    cols=['accuracy','sensitivity','specificity','balanced_accuracy','f1_score','gmean','epoch','dataset']
    pd.DataFrame(metrics_rows)[cols].to_excel('metrics/training_metrics.xlsx',index=False)

def _save_best_metrics(tag,datasets):
    rows=[]
    for name,(labels,preds) in datasets.items():
        m=_analysis_metrics(labels,preds); rows.append({'dataset':name,'accuracy':m['accuracy'],'sensitivity':m['sensitivity'],'specificity':m['specificity'],'balanced_acc':m['balanced_accuracy'],'f1':m['f1_score'],'gmean':m['gmean']})
    path=f'BEST_METRICS_{tag}.xlsx'; pd.DataFrame(rows).to_excel(path,index=False); print('Saved:',path)

def _save_best_analysis(best_classifier,best_diffusion,best_tag):
    os.makedirs('visualizations/confusion',exist_ok=True); os.makedirs('visualizations/tsne',exist_ok=True); os.makedirs('metrics',exist_ok=True); os.makedirs('diffusion_outputs',exist_ok=True)
    best_classifier.eval(); best_diffusion.eval()
    with torch.no_grad():
        x_best=sample(best_diffusion,args.batch_size,corrector_steps=4,snr=0.2); pseudo_best=torch.argmax(best_classifier(x_best),dim=1)
    datasets={}; analysis={}
    for name,loader in [('labeled',trainloader),('unlabeled',unlabelledloader),('test',testloader)]:
        feat,pred,lab=_collect_analysis_data(best_classifier,loader); datasets[name]=(lab,pred); analysis[name]=(feat,pred,lab)
    gf,gp,gl=_collect_generated_analysis_data(best_classifier,x_best,pseudo_best); datasets['generated']=(gl,gp); analysis['generated']=(gf,gp,gl)
    for name in ['labeled','unlabeled','test','generated']:
        feat,pred,lab=analysis[name]; _save_analysis_confusion(lab,pred,name,best_tag); _save_analysis_tsne(feat,lab,name,best_tag,1500 if name=='unlabeled' else 10000)
    _save_best_metrics(best_tag,datasets)
    x_vis=(x_best.clamp(-1,1)+1)/2; grid=torchvision.utils.make_grid(x_vis,nrow=4).clamp(0,1)
    plt.figure(figsize=(6,6)); plt.imshow(grid.permute(1,2,0).cpu()); plt.axis('off'); plt.savefig(os.path.join('diffusion_outputs',f'best_{best_tag}.png'),bbox_inches='tight'); plt.savefig(os.path.join('diffusion_outputs',f'best_{best_tag}.pdf'),bbox_inches='tight'); plt.close()

def main():
    

    set_all_seeds(10)
    global global_step
    global best_prec1
    global best_test_ema_prec1
    global best_acc

    
    print('| Building net type [' + args.arch + ']...')
    model = getNetwork(args, num_classes)
    ema_model = getNetwork(args, num_classes,ema=True)
    
    if use_cuda:
        model.cuda()
        ema_model.cuda()
        cudnn.benchmark = True

    if args.dataset != 'mnist' or args.dataset != 'fashion':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum)
    exp_name = experiment_name(sl = args.sl,
                    dataset= args.dataset,
                    labels = args.num_labeled,
                    valid = args.num_valid_samples,
                    optimizer = args.optimizer,
                    lr = args.lr,
                    init_lr = args.initial_lr,
                    lr_rampup = args.lr_rampup, 
                    lr_rampdown = args.lr_rampdown_epochs,
                    l2 = args.weight_decay,
                    ema_decay = args.ema_decay, 
                    mixup_consistency = args.mixup_consistency,
                    consistency_type = args.consistency_type,
                    consistency_rampup_s = args.consistency_rampup_starts,
                    consistency_rampup_e = args.consistency_rampup_ends,
                    epochs = args.epochs,
                    batch_size = args.batch_size,
                    mixup_sup_alpha = args.mixup_sup_alpha,
                    mixup_usup_alpha = args.mixup_usup_alpha,
                    mixup_hidden = args.mixup_hidden,
                    num_mix_layer = args.num_mix_layer,
                    pseudo_label = args.pseudo_label,
                    arch = args.arch,
                    dropout = args.dropout,
                    nesterov = args.nesterov,
                    job_id = args.job_id,
                    add_name= args.add_name)
    exp_name='s'


    exp_dir = args.root_dir+exp_name
    print (exp_dir)
    if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
    
    result_path = os.path.join(exp_dir , 'out.txt')
    if args.resume=='y':
        filep = open(result_path, 'a')
    else:
        filep = open(result_path, 'w')    
    out_str = str(args)
    filep.write(out_str + '\n')     
    
   
    
    if args.evaluate:
        print("Evaluating the primary model:\n")
        validate(validloader, model, global_step, args.start_epoch, filep)
        print("Evaluating the EMA model:\n")
        validate(validloader, ema_model, global_step, args.start_epoch, filep)
        return

    os.makedirs('metrics', exist_ok=True)
    os.makedirs('visualizations/confusion', exist_ok=True)
    os.makedirs('visualizations/tsne', exist_ok=True)
    os.makedirs('diffusion_outputs', exist_ok=True)
    if os.path.exists('metrics/training_metrics.xlsx'):
        os.remove('metrics/training_metrics.xlsx')
    metrics_rows.clear()
    best_acc=0.0; best_epoch=0; best_time=0.0; best_tag=None

    for epoch in range(args.start_epoch,args.epochs):
        start_time=time.time()
        if args.sl:
            x_diff,pseudo_idx=train_sl(trainloader,model,optimizer,epoch,filep)
        else:
            x_diff,pseudo_idx=train(trainloader,unlabelledloader,model,ema_model,optimizer,epoch,filep)
        eval_model=model if args.pseudo_label=='single' else ema_model
        val_acc=validate(validloader,eval_model,global_step,epoch+1,filep,ema=(args.pseudo_label!='single'),testing=False)
        cur_time=time.time()-start_time
        os.makedirs('acc_time',exist_ok=True)
        os.makedirs(os.path.join('acc_time','LUG'),exist_ok=True)
        index_tag = getattr(args, 'job_id', '') or '1'
        with open(os.path.join('acc_time','LUG',f'{args.num_labeled}_{index_tag}_{args.lr}.txt'),'a') as f: f.write(f'{cur_time} {float(val_acc)}\n')
        if float(val_acc)>best_acc:
            best_acc=float(val_acc); best_epoch=epoch+1; best_time=cur_time; best_tag=f'BEST_acc_{best_acc:.2f}_epoch_{best_epoch}'
            torch.save(eval_model.state_dict(),f'best_ema_classifier_{best_tag}.pth'); torch.save(model.state_dict(),f'best_classifier_{best_tag}.pth'); torch.save(ema_modeldif.state_dict(),f'best_diffusion_{best_tag}.pth')
            print('New BEST model saved:',best_tag)
        test_acc=validate(testloader,eval_model,global_step,epoch+1,filep,ema=(args.pseudo_label!='single'),testing=True)
        metric_model=eval_model
        _,pl,ll=_collect_analysis_data(metric_model,trainloader); _append_training_metric(epoch+1,'labeled',ll,pl)
        _,pu,lu=_collect_analysis_data(metric_model,unlabelledloader); _append_training_metric(epoch+1,'unlabeled',lu,pu)
        _,pt,lt=_collect_analysis_data(metric_model,testloader); _append_training_metric(epoch+1,'test',lt,pt)
        if x_diff is not None:
            _,pg,lg=_collect_generated_analysis_data(metric_model,x_diff,pseudo_idx); _append_training_metric(epoch+1,'generated',lg,pg)
        print('Validation accuracy: %.4f | Test accuracy: %.4f' % (float(val_acc),float(test_acc)))
        print('--- training epoch in %s seconds ---\n' % cur_time)

    if best_tag is None: best_tag=f'BEST_acc_{best_acc:.2f}_epoch_{best_epoch}'
    best_classifier_path=f'best_ema_classifier_{best_tag}.pth'; best_diffusion_path=f'best_diffusion_{best_tag}.pth'
    if os.path.exists(best_classifier_path):
        best_analysis_classifier=getNetwork(args,num_classes); best_analysis_classifier.cuda() if use_cuda else None; best_analysis_classifier.load_state_dict(torch.load(best_classifier_path,map_location=device))
    else: best_analysis_classifier=eval_model
    if os.path.exists(best_diffusion_path):
        best_analysis_diffusion=UNet(img_ch=3).to(device); best_analysis_diffusion.load_state_dict(torch.load(best_diffusion_path,map_location=device))
    else: best_analysis_diffusion=ema_modeldif
    _save_best_analysis(best_analysis_classifier,best_analysis_diffusion,best_tag)
    with open(os.path.join('acc_time','LUG',f'{args.num_labeled}_{args.lr}_best.txt'),'a') as f: f.write(f'{best_epoch} {best_time} {best_acc}\n')


    train_log = OrderedDict()
    train_log['train_class_loss_list'] = train_class_loss_list
    train_log['train_ema_class_loss_list'] = train_ema_class_loss_list
    train_log['train_mixup_consistency_loss_list'] = train_mixup_consistency_loss_list
    train_log['train_mixup_consistency_coeff_list'] = train_mixup_consistency_coeff_list
    train_log['train_error_list'] = train_error_list
    train_log['train_ema_error_list'] = train_ema_error_list
    train_log['train_lr_list'] = train_lr_list
    train_log['val_class_loss_list'] = val_class_loss_list
    train_log['val_error_list'] = val_error_list
    train_log['val_ema_class_loss_list'] = val_ema_class_loss_list
    train_log['val_ema_error_list'] = val_ema_error_list
    train_log['diffusion_loss_list'] = diffusion_loss_list
        
    filep.flush()
    pickle.dump(train_log, open( os.path.join(exp_dir,'log.pkl'), 'wb'))


def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train_sl(trainloader, model, optimizer, epoch, filep):
    global global_step
    
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    
    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    
    end = time.time()
    
    for i, (input, target)in enumerate(trainloader):
        # measure data loading time
        meters.update('data_time', time.time() - end)
        #if args.dataset == 'cifar10':
            #input = apply_zca(input, zca_mean, zca_components)
        
        
        lr = adjust_learning_rate(optimizer, epoch, i, len(unlabelledloader))
        meters.update('lr', optimizer.param_groups[0]['lr'])
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target.cuda())

        minibatch_size = len(target_var)
        #labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        #assert labeled_minibatch_size > 0
        
        model_out = model(input_var)

        logit1 = model_out
        class_logit, cons_logit = logit1, logit1
        
        class_loss = class_criterion(class_logit, target_var) / minibatch_size
        meters.update('class_loss', class_loss.item())

        
        loss = class_loss
        #assert not (np.isnan(loss.item()) or loss.item() > 1e5), 'Loss explosion: {}'.format(loss.data[0])
        assert not (np.isnan(loss.item())), 'Loss explosion: {}'.format(loss.data[0])
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 4))
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100. - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100. - prec5[0], minibatch_size)

        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        
        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()



            
    train_class_loss_list.append(meters['class_loss'].avg)
    train_error_list.append(meters['error1'].avg)
    train_lr_list.append(meters['lr'].avg)
    diffusion_loss_list.append(float('nan'))
    return None, None
    


def train(trainloader,unlabelledloader, model, ema_model, optimizer, epoch, filep):
    global global_step
    epoch_train_loss = 0.0
    train_loss_steps = 0
    epoch_diffusion_loss = 0.0
    diffusion_loss_steps = 0    
    class_criterion = nn.CrossEntropyLoss().cuda()
    criterion_u= nn.KLDivLoss(reduction='batchmean').cuda()
    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type
    
    meters = AverageMeterSet()

    # switch to train mode
    model.train()
    ema_model.train()

    end = time.time()
    i = -1

    labeled_iter = trainloader.__iter__()
    for o,(u, _) in enumerate(unlabelledloader):
        # measure data loading time
        i = i+1
        try:
              (input, target) = labeled_iter.__next__()
        except StopIteration:
              labeled_iter = trainloader.__iter__()
              (input, target)= labeled_iter.__next__()

            
        meters.update('data_time', time.time() - end)
        #print(input.shape)
        if input.shape[0]!= u.shape[0]:
            bt_size = np.minimum(input.shape[0], u.shape[0])
            input = input[0:bt_size]
            target = target[0:bt_size]
            u = u[0:bt_size]
        
        
        #if args.dataset == 'cifar10':
            #input = apply_zca(input, zca_mean, zca_components)
            #u = apply_zca(u, zca_mean, zca_components) 
        if args.dataset != 'mnist' or args.dataset != 'fashion':
            lr = adjust_learning_rate(optimizer, epoch, i, len(unlabelledloader))
            meters.update('lr', optimizer.param_groups[0]['lr'])
        
        if args.mixup_sup_alpha:
            if use_cuda:
                input , target, u  = input.cuda(), target.cuda(), u.cuda()
            input_var, target_var, u_var = Variable(input), Variable(target), Variable(u) 
            
            if args.mixup_hidden:
                output_mixed_l, target_a_var, target_b_var, lam = model(input_var, target_var, mixup_hidden = True,  mixup_alpha = args.mixup_sup_alpha, layers_mix = args.num_mix_layer)
                lam = lam[0]
            else:
                mixed_input, target_a, target_b, lam = mixup_data_sup(input, target, args.mixup_sup_alpha)
                #if use_cuda:
                #    mixed_input, target_a, target_b  = mixed_input.cuda(), target_a.cuda(), target_b.cuda()
                mixed_input_var, target_a_var, target_b_var = Variable(mixed_input), Variable(target_a), Variable(target_b)
                output_mixed_l = model(mixed_input_var)
                #print(output_mixed_l.shape)    
            loss_func = mixup_criterion(target_a_var, target_b_var, lam)
            class_loss = loss_func(class_criterion, output_mixed_l)
            
        else:
            input_var = torch.autograd.Variable(input.cuda())
            with torch.no_grad():
                u_var = torch.autograd.Variable(u.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            output = model(input_var)
            class_loss = class_criterion(output, target_var)
        
        meters.update('class_loss', class_loss.item())
        
        ### get ema loss. We use the actual samples(not the mixed up samples ) for calculating EMA loss
        minibatch_size = len(target_var)
        if args.pseudo_label == 'single':
            ema_logit_unlabeled = model(u_var)
            ema_logit_labeled = model(input_var)
        else:
            ema_logit_unlabeled = ema_model(u_var)
            ema_logit_labeled = ema_model(input_var)
        if args.mixup_sup_alpha:
            class_logit = model(input_var)
        else:
            class_logit = output
        cons_logit = model(u_var)

        ema_logit_unlabeled = Variable(ema_logit_unlabeled.detach().data, requires_grad=False)

        #class_loss = class_criterion(class_logit, target_var) / minibatch_size
        
        ema_class_loss = class_criterion(ema_logit_labeled, target_var)# / minibatch_size
        meters.update('ema_class_loss', ema_class_loss.item())
        
               
        ### get the unsupervised mixup loss###
        if args.mixup_consistency:
                if args.mixup_hidden:
                    #output_u = model(u_var)
                    output_mixed_u, target_a_var, target_b_var, lam = model(u_var, ema_logit_unlabeled, mixup_hidden = True,  mixup_alpha = args.mixup_sup_alpha, layers_mix = args.num_mix_layer)
                    # ema_logit_unlabeled
                    lam = lam[0]
                    mixedup_target = lam * target_a_var + (1 - lam) * target_b_var
                else:
                    #output_u = model(u_var)
                    mixedup_x, mixedup_target, lam = mixup_data(u_var, ema_logit_unlabeled, args.mixup_usup_alpha)
                    #mixedup_x, mixedup_target, lam = mixup_data(u_var, output_u, args.mixup_usup_alpha)
                    output_mixed_u = model(mixedup_x)
                mixup_consistency_loss = consistency_criterion(output_mixed_u, mixedup_target) / minibatch_size# criterion_u(F.log_softmax(output_mixed_u,1), F.softmax(mixedup_target,1))
                meters.update('mixup_cons_loss', mixup_consistency_loss.item())
                if epoch < args.consistency_rampup_starts:
                    mixup_consistency_weight = 0.0
                else:
                    mixup_consistency_weight = get_current_consistency_weight(args.mixup_consistency, epoch, i, len(unlabelledloader))
                meters.update('mixup_cons_weight', mixup_consistency_weight)
                mixup_consistency_loss = mixup_consistency_weight*mixup_consistency_loss
        else:
            mixup_consistency_loss = 0
            meters.update('mixup_cons_loss', 0)
        
        #labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        #assert labeled_minibatch_size > 0













        C_fake_pred= model(u_var)
        C_fake_pred = F.softmax(C_fake_pred, dim=1)


        C_fake_wei = torch.max(C_fake_pred, 1)[1]                              
        C_fake_wei = C_fake_wei.view(-1, 1)
        C_fake_wei = torch.zeros(C_fake_wei.size(0), 10).cuda().scatter_(1, C_fake_wei, 1)
        C_unlabeled_loss = nll_loss_neg2(C_fake_wei, C_fake_pred)

        #C_unlabeled_loss+








        loss = class_loss+mixup_consistency_loss#C_unlabeled_loss+0.1*C_fake_lossmix+C_labeled_loss +C_unlabeled_loss+generated_weight(epoch)*C_u_lossmix
        
        meters.update('loss', loss.item())

        prec1, prec5 = accuracy(class_logit.data, target_var.data, topk=(1, 4))
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100. - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100. - prec5[0], minibatch_size)

        ema_prec1, ema_prec5 = accuracy(ema_logit_labeled.data, target_var.data, topk=(1, 4))
        meters.update('ema_top1', ema_prec1[0], minibatch_size)
        meters.update('ema_error1', 100. - ema_prec1[0], minibatch_size)
        meters.update('ema_top5', ema_prec5[0], minibatch_size)
        meters.update('ema_error5', 100. - ema_prec5[0], minibatch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {meters[batch_time]:.3f}\t'
                'Data {meters[data_time]:.3f}\t'
                'Class {meters[class_loss]:.4f}\t'
                'Mixup Cons {meters[mixup_cons_loss]:.4f}\t'
                'Prec@1 {meters[top1]:.3f}\t'
                'Prec@5 {meters[top5]:.3f}'.format(
                    epoch, i, len(unlabelledloader), meters=meters))
            #print ('lr:',optimizer.param_groups[0]['lr'])



        # update Dfifusion network
        modeldif.train()
        optimizerdif.zero_grad()
        # loss (score matching)
        lossdif = loss_fn(modeldif, u_var)

        # backward
        lossdif.backward()
        optimizerdif.step()

        # EMA update
        update_ema(modeldif, ema_modeldif)        

        # update D network





    model.train()

    with torch.no_grad():

                # generate samples
                x_diff = sample(
                    ema_modeldif,
                    args.batch_size,
                    corrector_steps=4,
                    snr=0.2
                )

                # normalize
                x_vis = (x_diff.clamp(-1,1) + 1) / 2
                # save image
                grid = torchvision.utils.make_grid(x_vis, nrow=4)
                grid = grid.clamp(0,1)
                plt.figure(figsize=(6,6))
                plt.imshow(grid.permute(1,2,0).cpu())
                plt.axis("off")

                save_path = f"diffusion_outputs/epoch_{epoch+1}.png"
                pdf_path = f"diffusion_outputs/epoch_{epoch+1}.pdf"
                plt.savefig(save_path, bbox_inches="tight")
                plt.savefig(pdf_path, bbox_inches="tight")
                plt.close()

            # =========================
            # TRAIN CLASSIFIER ON
            # DIFFUSION SAMPLES
            # =========================

    optimizer.zero_grad()


    cnnlogits = model(x_diff)
    diff_probs = F.softmax(cnnlogits, dim=1)
    diff_logits=F.log_softmax(cnnlogits, dim=1)


    pseudo_idx = torch.argmax(
                diff_probs,
                dim=1
            )

    pseudo_onehot = torch.zeros(
                args.batch_size,
                10,
                device=device
            )

    pseudo_onehot.scatter_(
                1,
                pseudo_idx.view(-1,1),
                1
            )
    classifier_total_loss = nll_loss_neg3(
                diff_probs,
                pseudo_onehot
            )
    (0.1 * classifier_total_loss).backward()

    optimizer.step()

    update_ema_variables(model, ema_model, args.ema_decay, global_step)


    for p in model.parameters():
        p.requires_grad = False

    modeldif.train()
    optimizerdif.zero_grad()



            # --------------------------------
            # use REAL images as x0
            # --------------------------------
    x0 = x_diff   
    bs = x0.shape[0]
            # random timestep
    t = torch.rand(bs, device=device) * 0.8 + 0.2
    x0 = x0.to(device)
    t = t.to(device)
            # gaussian noise
    z = torch.randn_like(x0)

            # forward diffusion
    mean, std = sde.marginal_prob(x0, t)

    x_t = mean + std[:, None, None, None] * z

            # predict score
    score = modeldif(x_t, t)

            # one-step denoise estimate
    x_fake = x_t + (std[:, None, None, None] ** 2) * score

    x_fake = torch.tanh(x_fake)

            # --------------------------------
            # classifier prediction
            # IMPORTANT: NO no_grad HERE
            # --------------------------------
    cnnlogits = model(x_fake)
    diff_probs = F.softmax(cnnlogits, dim=1)
    diff_logits=F.log_softmax(cnnlogits, dim=1)
            # pseudo labels
    pseudo_labels = diff_probs.detach().argmax(dim=1)

            # confidence loss
    confidence_loss = F.nll_loss(
                diff_logits,
                pseudo_labels
            )

            # original score matching loss
    score_loss = torch.mean(
                (score * std[:, None, None, None] + z) ** 2
            )

            # total diffusion loss
    total_loss = score_loss + 0.05 * confidence_loss

            # Accumulate epoch-level DIFFUSION LOSS
    epoch_diffusion_loss += total_loss.item()
    diffusion_loss_steps += 1

    total_loss.backward()

    optimizerdif.step()

    update_ema(modeldif, ema_modeldif)

            # unfreeze classifier
    for p in model.parameters():
                p.requires_grad = True

    diffusion_loss_list.append(epoch_diffusion_loss / diffusion_loss_steps if diffusion_loss_steps else float("nan"))
    return x_diff.detach(), pseudo_labels.detach()


def visualize_results(G, epoch):
    G.eval()
    generated_images_dir = 'experiments/s/' + args.dataset
    if not os.path.exists(generated_images_dir):
        os.makedirs(generated_images_dir)

    tot_num_samples = 64
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    sample_z_ = torch.randn((tot_num_samples, z_dim))

    sample_z_ = sample_z_.cuda()

    samples = G(sample_z_)

    samples1=samples
    samples = samples.mul(0.5).add(0.5)

    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)


    save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      generated_images_dir + '/' + 'epoch%03d' % epoch + '.png')
    gridOfFakeImages = torchvision.utils.make_grid((samples1 + 1) / 2)
    torchvision.utils.save_image(gridOfFakeImages,generated_images_dir + '/' + 'epoch%03d' % epoch + '.pdf')


    


  

def validate(eval_loader, model, global_step, epoch, filep, ema = False, testing = False):
    class_criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=NO_LABEL).cuda()
    meters = AverageMeterSet()
    with torch.no_grad():
                visualize_results(G, (epoch + 1))
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(eval_loader):
        meters.update('data_time', time.time() - end)
    
        #if args.dataset == 'cifar10':
            #input = apply_zca(input, zca_mean, zca_components)
            
        with torch.no_grad():        
            input_var = torch.autograd.Variable(input.cuda())
        with torch.no_grad():
            target_var = torch.autograd.Variable(target.cuda())

        minibatch_size = len(target_var)
        labeled_minibatch_size = target_var.data.ne(NO_LABEL).sum().type(torch.cuda.FloatTensor)
        assert labeled_minibatch_size > 0
        meters.update('labeled_minibatch_size', labeled_minibatch_size)

        # compute output
        output1 = model(input_var)
        softmax1 = F.softmax(output1, dim=1)
        class_loss = class_criterion(output1, target_var) / minibatch_size

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output1.data, target_var.data, topk=(1, 4))
        meters.update('class_loss', class_loss.item(), minibatch_size)
        meters.update('top1', prec1[0], minibatch_size)
        meters.update('error1', 100.0 - prec1[0], minibatch_size)
        meters.update('top5', prec5[0], minibatch_size)
        meters.update('error5', 100.0 - prec5[0], minibatch_size)

        # measure elapsed time
        meters.update('batch_time', time.time() - end)
        end = time.time()
        
    print(' * Prec@1 {top1.avg:.3f}\tPrec@5 {top5.avg:.3f}\n'
          .format(top1=meters['top1'], top5=meters['top5']))

    
    if testing == False:
        if ema:
            val_ema_class_loss_list.append(meters['class_loss'].avg)
            val_ema_error_list.append(meters['error1'].avg)
        else:
            val_class_loss_list.append(meters['class_loss'].avg)
            val_error_list.append(meters['error1'].avg)
    
    
    return meters['top1'].avg


def save_checkpoint(state, is_best, dirpath, epoch):
    filename = 'checkpoint.{}.ckpt'.format(epoch)
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    torch.save(state, checkpoint_path)
    print("--- checkpoint saved to %s ---" % checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        print
        ("--- checkpoint copied to %s ---" % best_path)


def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    lr = ramps.linear_rampup(epoch, args.lr_rampup) * (args.lr - args.initial_lr) + args.initial_lr

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= ramps.cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr

def adjust_learning_rate_step(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_current_consistency_weight(final_consistency_weight, epoch, step_in_epoch, total_steps_in_epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    epoch = epoch - args.consistency_rampup_starts
    epoch = epoch + step_in_epoch / total_steps_in_epoch
    return final_consistency_weight * ramps.sigmoid_rampup(epoch, args.consistency_rampup_ends - args.consistency_rampup_starts )


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    #labeled_minibatch_size = max(target.ne(NO_LABEL).sum(), 1e-8).type(torch.cuda.FloatTensor)
    minibatch_size = len(target)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / minibatch_size))
    return res


def save(cc,save_dir, ep):

    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    torch.save(G.state_dict(), os.path.join(save_dir, '_G.pkl'))
    torch.save(D.state_dict(), os.path.join(save_dir, '_D.pkl'))
    torch.save(cc.state_dict(), os.path.join(save_dir,  '_C.pkl'))


# =========================
# 11. PREDICTOR-CORRECTOR SAMPLER
# =========================
@torch.no_grad()
def sample(model, n_samples=16, corrector_steps=4, snr=0.2):
    model.eval()
    x = torch.randn(n_samples, 3, image_size, image_size).to(device)

    timesteps = torch.linspace(1, 1e-3, sde.N).to(device)

    for i in range(len(timesteps) - 1):
        t = timesteps[i]
        t_next = timesteps[i + 1]

        vec_t = torch.ones(n_samples, device=device) * t
        vec_t_next = torch.ones(n_samples, device=device) * t_next

        # -------- Predictor --------
        g = sde.diffusion_coeff(vec_t)
        score = model(x, vec_t)

        dt = t_next - t

        x_mean = x + (g[:, None, None, None] ** 2) * score * dt
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(torch.abs(g[:, None, None, None] ** 2 * dt)) * noise

        # -------- Corrector --------
        for _ in range(corrector_steps):
            grad = model(x, vec_t_next)
            noise = torch.randn_like(x)

            grad_norm = torch.norm(
                grad.reshape(grad.shape[0], -1), dim=-1
            ).mean()

            noise_norm = torch.norm(
                noise.reshape(noise.shape[0], -1), dim=-1
            ).mean()

            step_size = (snr * noise_norm / (grad_norm + 1e-6)) ** 2 * 2

            x = x + step_size * grad
            x = x + torch.sqrt(2 * step_size) * noise

    return x
        


if __name__ == '__main__':
     main()