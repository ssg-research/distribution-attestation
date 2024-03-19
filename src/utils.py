# Authors: Vasisht Duddu, Anudeep Das, Nora Khayata, Hossein Yalame, Thomas Schneider, N Asokan
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tqdm import tqdm
import torch
import numpy as np
from copy import deepcopy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
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


def train(model, loaders, lr=1e-3, epoch_num=10,weight_decay=0, get_best=False):
    train_loader, val_loader = loaders
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)

    iterator = tqdm(range(1, epoch_num+1))
    best_model, best_loss = None, np.inf
    for epoch in iterator:
        _, tacc = train_epoch(train_loader, model,criterion, optimizer, epoch)
        vloss, vacc = validate_epoch(val_loader, model, criterion)
        iterator.set_description("train_acc: %.2f | val_acc: %.2f |" % (tacc, vacc))
        if get_best and vloss < best_loss:
            best_loss = vloss
            best_model = deepcopy(model)
    if get_best:
        return best_model, (vloss, vacc)
    return vloss, vacc

def train_epoch(train_loader, model, criterion, optimizer, epoch):
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    iterator = train_loader
    iterator = tqdm(train_loader)
    for data in iterator:
        images, labels, _ = data
        images, labels = images.to(device), labels.to(device)
        N = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)[:, 0]

        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        prediction = (outputs >= 0)
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
        train_loss.update(loss.item())

        iterator.set_description('[Train] Epoch %d, Loss: %.5f, Acc: %.4f' % (epoch, train_loss.avg, train_acc.avg))
    return train_loss.avg, train_acc.avg


def validate_epoch(val_loader, model, criterion, verbose = True, return_preds = False):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    preds_to_ret = []
    with torch.no_grad():
        for data in val_loader:
            images, labels, _ = data
            images, labels = images.to(device), labels.to(device)
            N = images.size(0)
            outputs = model(images)[:, 0]
            prediction = (outputs >= 0)
            if return_preds:
                preds_to_ret.append(prediction)
            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
            val_loss.update(criterion(outputs, labels.float()).item())
    if verbose:
        print('[Validation], Loss: %.5f, Accuracy: %.4f' %(val_loss.avg, val_acc.avg))
    if return_preds:
        return val_loss.avg, val_acc.avg, preds_to_ret
    return val_loss.avg, val_acc.avg
