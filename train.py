import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.optim as optim
import torch.nn as nn
from dataset import load_ddi_dataset
from logs.train_logger import TrainLogger
from data_preprocessing import CustomData
import argparse
from metrics import *
from utils import *
from tqdm import tqdm
import warnings
from model import DDI
warnings.filterwarnings("ignore")
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
set_seed(42)


if torch.cuda.is_available():
    device = torch.device('cuda')  
else:
    device = torch.device('cpu')


def val(model, criterion, dataloader, device, epoch):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='val_epoch_{}'.format(epoch), leave=True):
            data = [d.to(device) for d in data]
            head_atom, tail_atom, head_motif, tail_motif, motif_bipart, label = data

            label = label.long()
            pred = model(head_atom, tail_atom, head_motif, tail_motif, motif_bipart)
            loss = criterion(pred, label)

            pred_cls = torch.softmax(pred, dim=1)
            pred_list.append(pred_cls)
            label_list.append(label)
            running_loss.update(loss.item(), label.size(0))

    pred_logits = torch.cat(pred_list).view(-1, 4).cpu().numpy() 
    labels = torch.cat(label_list).view(-1).cpu().numpy()

    metrics = do_compute_metrics(pred_logits, labels)
    epoch_loss = running_loss.get_average()
    running_loss.reset()

    model.train()
    return epoch_loss, *metrics  

def test(model, criterion, dataloader, device):
    model.eval()
    running_loss = AverageMeter()

    pred_list = []
    label_list = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc='Testing', leave=True):
            data = [d.to(device) for d in data]
            head_atom, tail_atom, head_motif, tail_motif, motif_bipart, label = data

            label = label.long()

            pred = model(head_atom, tail_atom, head_motif, tail_motif, motif_bipart)
            loss = criterion(pred, label)
            pred_cls = torch.softmax(pred, dim=1)
            pred_list.append(pred_cls)
            label_list.append(label)
            running_loss.update(loss.item(), label.size(0))

    pred_logits = torch.cat(pred_list).view(-1, 4).cpu().numpy()
    labels = torch.cat(label_list).view(-1).cpu().numpy()

    metrics = do_compute_metrics(pred_logits, labels)
    epoch_loss = running_loss.get_average()
    return epoch_loss, *metrics


def main():
    train_loader, val_loader, test_loader = load_ddi_dataset(root='data')
    model = DDI(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** epoch)

    running_loss = AverageMeter()
    running_acc = AverageMeter()

    model.train()
    for epoch in range(epochs):
        running_loss.reset()
        running_acc.reset()

        for data in tqdm(train_loader, desc='train_loader_epoch_{}'.format(epoch), leave=True):
            data = [d.to(device) for d in data]
            head_atom, tail_atom, head_motif, tail_motif, motif_bipart, label = data
            pred = model.forward(head_atom, tail_atom, head_motif, tail_motif, motif_bipart)
            pred = pred.float()
            label = label.long()
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pred_cls = torch.argmax(pred, dim=1)
            running_acc.update((pred_cls == label).float().mean().item(), label.size(0))
            running_loss.update(loss.item(), label.size(0))
        epoch_loss = running_loss.get_average()
        epoch_acc = running_acc.get_average()

        val_metrics = val(model, criterion, val_loader, device, epoch)
        if val_metrics[1] > max_acc:
            max_acc = val_metrics[1]
            torch.save(model.state_dict(), pkl_name)
        scheduler.step()
    torch.save(model.state_dict(), last_name)


if __name__ == "__main__":
    main()
