from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pytz
from tensorboardX import SummaryWriter
import math
import tqdm
import socket
from utils.metrics import *
from utils.Utils import *
from .fourier import fourier_amplitude_mix

import copy
from torch.distributions.uniform import Uniform
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
softmax = torch.nn.Softmax(-1)
celoss = nn.CrossEntropyLoss()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class InfoNCE(torch.nn.Module):

    def __init__(self, temperature=1.0):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, zis, zjs):
        batch_size, C = zis.size()[:2]
        zis = zis.contiguous().view(batch_size, C, -1).mean(dim=2).view(batch_size, C)
        zjs = zjs.contiguous().view(batch_size, C, -1).mean(dim=2).view(batch_size, C)

        rep = torch.cat([zjs, zis], dim=0)
        rep = torch.nn.functional.normalize(rep)
        similarity_matrix = self.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(0))
        exp_cosine_matrix = torch.exp(similarity_matrix / self.temperature)

        logit_i = torch.log((exp_cosine_matrix[:batch_size, :batch_size]) /
                          (exp_cosine_matrix[:batch_size, :batch_size] + (exp_cosine_matrix[:batch_size, batch_size:]).sum(dim=1)))

        logit_j = torch.log((exp_cosine_matrix[batch_size:, batch_size:]) /
                          (exp_cosine_matrix[batch_size:, batch_size:] + (exp_cosine_matrix[batch_size:, :batch_size]).sum(dim=1)))
        return logit_i.mean() + logit_j.mean()


class Trainer(object):

    def __init__(self, cuda, model, lr, val_loader, train_loader, out, max_epoch, optim, stop_epoch=None,
                 lr_decrease_rate=0.1, interval_validate=None, batch_size=8, gam=200, action=None):
        self.cuda = cuda
        self.model = model

        self.optim = optim
        self.lr = lr
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(5)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)
        self.infonce = InfoNCE()
        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'valid/cup_dice',
            'valid/disc_dice',
            'valid/loss_CE',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.total_iters = self.stop_epoch * len(self.train_loader)
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1


        self.temperature = 10.0
        self.gamma = gam
        self.m = 0.9995
        self.w = 1


    def validate(self):
        training = self.model.training
        self.model.eval()

        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        metrics = []
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):

                image = sample['image']
                label = sample['label']
                domain_code = sample['dc']
                index = sample['img_idx']

                data = image.to(device)
                target_map = label.to(device)
                domain_code = domain_code.to(device)
                index = index.to(device)

                with torch.no_grad():
                    predictions = self.model(data, mode='val')

                loss_seg = bceloss(torch.sigmoid(predictions), target_map)
                loss_data = loss_seg.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                dice_cup, dice_disc = dice_coeff_2label(np.asarray(torch.sigmoid(predictions.data.cpu())) > 0.75, target_map)
                val_cup_dice += dice_cup
                val_disc_dice += dice_disc
            val_loss /= len(self.val_loader)
            val_cup_dice /= len(self.val_loader)
            val_disc_dice /= len(self.val_loader)
            metrics.append((val_loss, val_cup_dice, val_disc_dice))
            self.writer.add_scalar('val_data/loss', val_loss, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.train_loader)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.train_loader)))

            mean_dice = val_cup_dice + val_disc_dice
            record_str = "\n[Epoch: {:d}] val CUP dice: {:f}, val DISC dice: {:f}, val Loss: {:.5f}".format(self.epoch + 1, val_cup_dice, val_disc_dice, val_loss)

            print(record_str)
            is_best = mean_dice > self.best_mean_dice

            with open(osp.join(self.out, 'log.txt'), 'a') as f:
                f.write(record_str)

            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model.__class__.__name__,
                    'optim_state_dict': self.optim.state_dict(),
                    'model_state_dict': self.model.state_dict(),
                    'learning_rate_gen': get_lr(self.optim),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d_best.pth.tar' % self.best_epoch))
            else:
                if (self.epoch + 1) % 100 == 0 or (self.epoch + 1) % 50 == 0:
                    torch.save({
                        'epoch': self.epoch,
                        'iteration': self.iteration,
                        'arch': self.model.__class__.__name__,
                        'optim_state_dict': self.optim.state_dict(),
                        'model_state_dict': self.model.state_dict(),
                        'learning_rate_gen': get_lr(self.optim),
                        'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))

            if training:
                self.model.train()

    def train_epoch(self):
        self.model.train()
        self.running_seg_loss = 0.0
        self.running_cls_loss = 0
        start_time = timeit.default_timer()

        for batch_idx, sample in tqdm.tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc='Train epoch=%d' % (self.epoch+1), ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration

            assert self.model.training
            self.optim.zero_grad()

            image = None
            label = None
            domain_code = None
            index = None
            for domain in sample:
                if image is None:
                    image = domain['image']
                    label = domain['label']
                    domain_code = domain['dc']
                    index = domain['img_idx']

                else:
                    image = torch.cat([image, domain['image']], 0)
                    label = torch.cat([label, domain['label']], 0)
                    domain_code = torch.cat([domain_code, domain['dc']], 0)
                    index = torch.cat([index, domain['img_idx']], 0)

            image = image.to(device)
            target_map = label.to(device)
            domain_code = domain_code.to(device)
            b = image.shape[0] // 6
            image = torch.cat([image[:b], image[2*b:3*b], image[4*b:5*b], image[b:2*b], image[3*b:4*b], image[5*b:6*b]], dim=0)
            target_map = torch.cat([target_map[:b], target_map[2*b:3*b], target_map[4*b:5*b], target_map[b:2*b], target_map[3*b:4*b], target_map[5*b:6*b]], dim=0)
            domain_code = torch.cat([domain_code[:b], domain_code[2*b:3*b], domain_code[4*b:5*b], domain_code[b:2*b], domain_code[3*b:4*b], domain_code[5*b:6*b]], dim=0)
            # domain_code = torch.nn.functional.one_hot(domain_code)

            out, image_style, d_pre, d_aft = self.model(image, mode='train')

            loss_seg = bceloss(torch.sigmoid(out[:3*b]), target_map[:3*b])
            loss_seg_2 = bceloss(torch.sigmoid(out[3*b:]), target_map[3*b:])
            loss_m = mseloss(image_style, image[3*b:])
            loss_pre = celoss(torch.softmax(d_pre, dim=1), domain_code[:3*b])
            loss_aft = celoss(torch.softmax(d_pre,dim=1), domain_code[3*b:])

            self.running_seg_loss += loss_seg.item()
            loss_data = (loss_seg + loss_seg_2 + 0.1*loss_m + 0.1*loss_pre).data.item()

            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            beta = ((self.epoch + 1) / 120) ** 0.9
            if loss_aft > loss_pre:
                loss_aft = torch.clamp(loss_aft, 0, loss_pre.item())

            loss = loss_seg + loss_seg_2 + 0.1*loss_m + 0.1*beta*loss_pre - 0.1*beta*loss_aft

            loss.backward()
            self.optim.step()

            # write image log
            if iteration % 30 == 0:
                grid_image = make_grid(
                    image[0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/image', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 0, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target_cup', grid_image, iteration)
                grid_image = make_grid(
                    target_map[0, 1, ...].clone().cpu().data, 1, normalize=True)
                self.writer.add_image('train/target_disc', grid_image, iteration)


            # write loss log
            self.writer.add_scalar('train_gen/loss', loss_data, iteration)
            self.writer.add_scalar('train_gen/loss_seg', loss_seg.data.item(), iteration)
        self.running_seg_loss /= len(self.train_loader)
        self.running_cls_loss /= len(self.train_loader)
        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, Execution time: %.5f' %
              (self.epoch+1, get_lr(self.optim), self.running_seg_loss, stop_time - start_time))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            # torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch + 1) % (self.max_epoch//2) == 0:
                _lr_gen = self.lr * self.lr_decrease_rate
                for param_group in self.optim.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr', get_lr(self.optim), self.epoch * (len(self.train_loader)))
            if (self.epoch + 1) % self.interval_validate == 0 or self.epoch == 0:
                self.validate()
        self.writer.close()

    def get_current_consistency_weight(self, epoch, consistency, consistency_rampup):
        return consistency * self.sigmoid_rampup(epoch, consistency_rampup)

    def sigmoid_rampup(self, current, rampup_length):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return (np.exp(-0.5 * phase * phase))




