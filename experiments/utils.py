from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import torch
from torch import nn

class HingeLoss(nn.Module):
    def __init__(self, device='cpu'):
        super(HingeLoss, self).__init__()
        self.device = device

    def forward(self, output, target):
        one_hot_mask = torch.zeros(len(target), output.shape[1]).to(self.device)
        one_hot_mask.scatter_(1, target.unsqueeze(1), 1.)
        loss = torch.sum((1-torch.diag(output[:,target])).clamp(min=0))+torch.sum(((1+output)*(1-one_hot_mask)).clamp(min=0))/10
        return loss/target.shape[0]

def get_data(datapath, dataset, num_train):
    if dataset == 'Cifar100':
        trainset = dset.CIFAR100(root=datapath, train=True, download=True)
    else:
        trainset = dset.CIFAR10(root=datapath, train=True, download=True)
    train_dat = (trainset.data.transpose(0,3,1,2)/255-0.5)/0.5
    train_tar = np.array(trainset.targets)

    if dataset == 'Cifar100':
        testset = dset.CIFAR100(root=datapath, train=False, download=True)
    else:
        testset = dset.CIFAR10(root=datapath, train=False, download=True)
    test_dat = (testset.data.transpose(0,3,1,2)/255-0.5)/0.5
    test_tar = np.array(testset.targets)
    
    return train_dat[:num_train], train_tar[:num_train], train_dat[num_train:], train_tar[num_train:], test_dat, test_tar

def train_model(data, fix, model, pars, ep_loss, ep_acc):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    device=pars.device
    dtype = torch.float32
    train_dat=data[0]; train_tar=data[1]
    val_dat=data[2]; val_tar=data[3]
    fix = fix.to(device=device)
    model = model.to(device=device)  # move the model parameters to CPU/GPU
       
    if pars.process == 'E2E':
      params=list(fix.parameters())+list(model.parameters())
    else:
      params=model.parameters()
    if pars.OPT=='SGD':
      optimizer = torch.optim.SGD(params, lr=pars.LR)
    else:
      optimizer = torch.optim.Adam(params, lr=pars.LR)

    for e in range(pars.epochs):
        running_loss = 0
        for j in np.arange(0,len(train_tar),pars.batch_size):

            model.train()  # put model to training mode
            x = torch.from_numpy(train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = torch.from_numpy(train_tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
            if pars.architecture == 'GLL':
              with torch.no_grad():
                  x1 = fix(x)
            else: # 'E2E'
              x1=fix(x)
            scores = model(x1)
            loss = pars.criterion(scores, y)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        running_loss /= (len(train_tar)/pars.batch_size)
        acc = check_accuracy(val_dat, val_tar, fix, model, pars)
        print('Epoch %d, loss = %.4f, val.acc = %.4f' % (e, running_loss, acc))

        ep_loss.append(running_loss)
        ep_acc.append(acc)


def check_accuracy(dat, tar, fix, model, pars):
        
    device=pars.device
    
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for j in np.arange(0,len(tar),pars.batch_size):
            x = torch.from_numpy(dat[j:j+pars.batch_size]).to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = torch.from_numpy(tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
            x1 = fix(x)
            scores = model(x1)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        #print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def train_model_rand(data, net, classifier, pars, ep_loss, ep_acc):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    device=pars.device
    dtype = torch.float32
    train_dat=data[0]; train_tar=data[1]
    val_dat=data[2]; val_tar=data[3]
    net = net.to(device=device)  # move the model parameters to CPU/GPU
    classifier = classifier.to(device=device)
    epochs = pars.epochs * pars.NUM_LAYER
    for e in range(epochs):
        running_loss = 0
        for j in np.arange(0,len(train_tar), pars.batch_size):
            choose_layer = torch.randint(0, pars.NUM_LAYER, (1,)).item()
            fix = net[:choose_layer]
            model = nn.Sequential(
                net[choose_layer],
                classifier[choose_layer]
            )

            optimizer = torch.optim.SGD(model.parameters(), pars.LR)

            model.train()  # put model to training mode
            x = torch.from_numpy(train_dat[j:j+pars.batch_size]).to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = torch.from_numpy(train_tar[j:j+pars.batch_size]).to(device=device, dtype=torch.long)
            
            with torch.no_grad():
                x1 = fix(x)
            scores = model(x1)
            loss = pars.criterion(scores, y)
            running_loss += loss.item()
            #print('Layer:{}, loss:{:.4f}'.format(choose_layer, loss.item()))

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

        running_loss /= (len(train_tar)/pars.batch_size)
        ep_loss.append(running_loss)        
        acc = check_accuracy_rand(val_dat, val_tar, net, classifier, pars)
        ep_acc.append(acc)
        print('Epoch {:d}, loss = {:.4f}, val.acc = {}'.format(e, running_loss, [round(x,4) for x in acc]))       


def check_accuracy_rand(dat, tar, net, classifier, pars):

    all_acc = []

    for i in range(0, pars.NUM_LAYER):
        fix = net[:i]
        model = nn.Sequential(
            net[i],
            classifier[i]
        )
        acc = check_accuracy(dat, tar, fix, model, pars)
        all_acc.append(acc)

    return all_acc