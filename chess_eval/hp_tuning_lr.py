#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader
import optuna
import mlflow
import random

def prep_data(path):
    import json
    df = np.load(path)

    if 'train' in path:
        ratings_json = {'min_rating':df[:,-3:-1].min(), 'max_rating':df[:,-3:-1].max()}
        with open('scaling.json','w') as f:
            json.dump(ratings_json,f)

    with open('scaling.json', 'r') as f:
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
    
    return val_loss, val_accuracy

class Network(nn.Module):
    def __init__(self, input_size, output_layer1, output_layer2):
        super(Network, self).__init__()

        self.linear1 = nn.Linear(input_size, output_layer1)
        self.batchnorm1 = nn.BatchNorm1d(num_features=output_layer1)

        self.linear2 = nn.Linear(output_layer1, output_layer2)
        self.batchnorm2 = nn.BatchNorm1d(num_features=output_layer2)

        self.linear3 = nn.Linear(output_layer2, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = nn.ReLU()(x)
        x = self.batchnorm1(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear2(x)
        x = nn.ReLU()(x)
        x = self.batchnorm2(x)
        x = nn.Dropout(0.5)(x)

        x = self.linear3(x)
        x = self.softmax(x)
        return x


def main():
    torch.manual_seed(123)
    random.seed(123)
    np.random.seed(123)

    device = torch.device("cpu")

    X_train,y_train = prep_data('data/train.npy')
    X_val, y_val = prep_data('data/val.npy')
    
    input_size = len(X_train[0,:])
    output_layer1= 32
    output_layer2 = 16

    model = Network(input_size=input_size, output_layer1=output_layer1, output_layer2=output_layer2)


    def objective(trial):
        with mlflow.start_run():
            params = {'epochs':trial.suggest_categorical('epochs', [10,50,100]),
                    'lr':trial.suggest_float('learning_rate',1e-4,0.8,log=True),
                    'gamma': trial.suggest_float('gamma',0.1,0.9),
                    'scheduler':trial.suggest_categorical('scheduler',['step','exponential']),
                    'batch_size':trial.suggest_categorical('batch_size',[128,256,512]),
                    'step_size':trial.suggest_int('step_size',5,20,5),
                    'optimizer': trial.suggest_categorical('optimiser',['SGD', 'Adam'])}
            mlflow.log_params(params)

            if params['optimizer'] == 'SGD':
                optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'])
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr = params['lr'])

            criterion_train = torch.nn.CrossEntropyLoss()
            criterion_test = torch.nn.CrossEntropyLoss(reduction='sum')

            if params['scheduler'] == 'exponential':
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma = params['gamma'])
            else:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=params['step_size'], gamma = params['gamma'])


            X_train_dataloader = DataLoader(X_train[:,:], batch_size=params['batch_size'])
            y_train_dataloader = DataLoader(y_train[:,:], batch_size=params['batch_size'])
            X_val_dataloader = DataLoader(X_val[:,:], batch_size=params['batch_size'])
            y_val_dataloader = DataLoader(y_val[:,:], batch_size=params['batch_size'])

            H = {'train_loss':[], 'val_loss':[], 'train_accuracy':[], 'val_accuracy':[]}
            for epoch in range(1, params['epochs'] + 1):
                    train_loss, train_accuracy = train(params, model, device, X_train_dataloader, y_train_dataloader, optimizer, criterion_train, criterion_test, epoch)
                    val_loss, val_accuracy = validate(model, device, X_val_dataloader, y_val_dataloader, criterion_test)
                    
                    H['train_loss'].append(train_loss)
                    H['train_accuracy'].append(train_accuracy)
                    H['val_loss'].append(val_loss)
                    H['val_accuracy'].append(val_accuracy)

                    scheduler.step()

            mlflow.log_metric('train_loss', H['train_loss'][-1])
            mlflow.log_metric('train_accuracy', H['train_accuracy'][-1])
            mlflow.log_metric('val_loss', H['val_loss'][-1])
            mlflow.log_metric('val_accuracy', H['val_accuracy'][-1])

            plt.style.use('ggplot')
            fig,ax = plt.subplots()
            ax.plot(np.arange(0,len(H['train_loss'])), H['train_loss'], label = 'train loss')
            ax.plot(np.arange(0,len(H['val_loss'])), H['val_loss'], label = 'val loss')
            ax.plot(np.arange(0,len(H['train_accuracy'])), H['train_accuracy'], label = 'train accuracy')
            ax.plot(np.arange(0,len(H['val_accuracy'])), H['val_accuracy'], label = 'val accuracy')
            ax.set_title('Training and Validation Loss and Accuracy')
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('Loss/Accuracy')
            ax.legend()

            mlflow.log_figure(fig, 'accuracy_loss_plot.png')
            
            mlflow.pytorch.log_model(model,'model')

        return val_accuracy 
    
    tracking_uri = r'./mlruns'
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('chess_eval')
    
    sampler = optuna.samplers.TPESampler(seed=123)
    study = optuna.create_study(direction='maximize',sampler=sampler)
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=50)
    


if __name__ =='__main__':
    main()
# %%