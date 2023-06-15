#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
from sklearn.utils import class_weight
from networks import Conv

def prep_data(path):
    import json
    df = np.load(path)

    if 'train' in path:
        ratings_json = {'min_rating':df[:,-3:-1].min(), 'max_rating':df[:,-3:-1].max()}
        with open('models/scaling.json','w') as f:
            json.dump(ratings_json,f)

    with open('models/scaling.json', 'r') as f:
        ratings_json = json.load(f)

    df[:,-3:-1] = (df[:,-3:-1] - ratings_json['min_rating']) / (ratings_json['max_rating'] - ratings_json['min_rating'])
    np.random.shuffle(df)

    X = torch.tensor(df[:,:-1])
    y = torch.tensor(df[:,-1])

    y = torch.nn.functional.one_hot(y.to(torch.int64))

    X = X.to(torch.float32)
    y= y.to(torch.float32)
    return X,y


def train(args, model, device, X_train_dataloader, y_train_dataloader, optimizer, criterion1, criterion2, epoch):
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(zip(X_train_dataloader, y_train_dataloader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion1(output, target)
        train_loss+=criterion2(output,target).item()
        loss.backward()
        optimizer.step()

        pred_index = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        target_index = target.argmax(dim=1, keepdim=True)
        correct += (pred_index == target_index).sum().item()
    
        # if batch_idx % args['log_interval'] == 0:
        #     print(f'Epoch: {epoch} \tLoss: {loss.item():.6f} \
        #           \t({batch_idx * len(data)}/{len(X_train_dataloader.dataset)})')
    
    train_loss /= len(X_train_dataloader.dataset)
    train_accuracy = correct / len(X_train_dataloader.dataset)

    print(f'Train set: \tAverage loss:\t{train_loss:.4f} \
          \tAccuracy: {correct}/{len(X_train_dataloader.dataset)} ({100. * train_accuracy:.1f}%)')
    return train_loss, train_accuracy

def validate(model, device, X_test_loader, y_test_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in zip(X_test_loader, y_test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()  # sum up batch loss
            pred_index = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            target_index = target.argmax(dim=1, keepdim=True)
            correct += (pred_index == target_index).sum().item()

    val_loss /= len(X_test_loader.dataset)
    val_accuracy = correct / len(X_test_loader.dataset)
    print(f'Val set: \tAverage loss:\t{val_loss:.4f} \
          \tAccuracy: {correct}/{len(X_test_loader.dataset)} ({100. * val_accuracy:.1f}%)')
    print("\n-----------------------------------------\n")
    
    return val_loss, val_accuracy


def main():
    try:
        ap = argparse.ArgumentParser()
        ap.add_argument('--epochs', type=int, default=14, metavar='N',
                            help='number of epochs to train (default: 14)')
        ap.add_argument('--lr', type=float, default=1.0, metavar='LR',
                            help='learning rate (default: 1.0)')
        ap.add_argument('--gamma', type=float, default=0.9, metavar='M',
                            help='Learning rate step gamma (default: 0.7)')
        ap.add_argument('--log-interval', type=int, default=1000, metavar='N',
                            help='how many batches to wait before logging training status')
        ap.add_argument('--scheduler', type=str, default='step', metavar = 'S',
                            help='learning rate scheduler to use, step or exponential')
        ap.add_argument('--batch-size', type=int, default=256, metavar='N',
                            help='input batch size for training (default: 64)')
        ap.add_argument('--save-model', type=str, default=False, metavar='S',
                            help='For Saving the current Model')        
        ap.add_argument('--step-size', type=int, default=10,
                                    help='Step size for lr scheduler')   
        args = vars(ap.parse_args())
    
    except:
        args = {'epochs': 100,
                'lr': 0.1,
                'gamma': 0.1,
                'log_interval': 10000
                }

    device = torch.device("cpu")

    X_train,y_train = prep_data('data/train.npy')
    X_val, y_val = prep_data('data/val2.npy')

    
    input_size = len(X_train[0,:])
    output_layer1= 128
    output_layer2 = 64

    # model = Network_2h(input_size, output_layer1, output_layer2)
    model = Conv()
    

    optimizer = torch.optim.SGD(model.parameters(), lr=args['lr'])
    criterion_train = torch.nn.CrossEntropyLoss()
    criterion_test = torch.nn.CrossEntropyLoss(reduction='sum')
    if args['scheduler'] == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = args['gamma'], verbose = True)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args['step_size'], gamma = args['gamma'], verbose = True)


    X_train_dataloader = DataLoader(X_train[:,:], batch_size=args['batch_size'])
    y_train_dataloader = DataLoader(y_train[:,:], batch_size=args['batch_size'])
    X_val_dataloader = DataLoader(X_val[:,:], batch_size=args['batch_size'])
    y_val_dataloader = DataLoader(y_val[:,:], batch_size=args['batch_size'])

    H = {'train_loss':[], 'val_loss':[], 'train_accuracy':[], 'val_accuracy':[]}
    for epoch in range(1, args['epochs'] + 1):
            print(f'Epoch: {epoch}')
            train_loss, train_accuracy = train(args, model, device, X_train_dataloader, y_train_dataloader, optimizer, criterion_train, criterion_test, epoch)
            val_loss, val_accuracy = validate(model, device, X_val_dataloader, y_val_dataloader, criterion_test)
            

            H['train_loss'].append(train_loss)
            H['train_accuracy'].append(train_accuracy)
            H['val_loss'].append(val_loss)
            H['val_accuracy'].append(val_accuracy)

            # scheduler.step()

            if epoch%5 == 0:
                plt.style.use('ggplot')
                plt.figure()
                plt.plot(np.arange(0,len(H['train_loss'])), H['train_loss'], label = 'train loss')
                plt.plot(np.arange(0,len(H['val_loss'])), H['val_loss'], label = 'val loss')
                plt.plot(np.arange(0,len(H['train_accuracy'])), H['train_accuracy'], label = 'train accuracy')
                plt.plot(np.arange(0,len(H['val_accuracy'])), H['val_accuracy'], label = 'val accuracy')
                plt.title('Training and Validation Loss and Accuracy')
                plt.xlabel('Epoch #')
                plt.ylabel('Loss/Accuracy')
                plt.legend()
                plt.savefig(f'figures/epochs_{args["epochs"]}__lr_{args["lr"]}__gamma_{args["gamma"]}__scheduler_{args["scheduler"]}.png')
            
    if args['save_model']:
        torch.save(model.state_dict(), "models/chess_model.pt")

if __name__ =='__main__':
    main()
#  python train.py --epochs 65 --lr 0.1 --gamma 0.8 --schedular step --step-size 15# %%