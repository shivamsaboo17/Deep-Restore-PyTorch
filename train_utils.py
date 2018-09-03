import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data import NoisyDataset
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from torch.utils.data import DataLoader


class Train:
    
    def __init__(self, architecture, train_dir, val_dir, params):
        
        self.architecture = architecture.cuda()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.noise_model = params['noise_model']
        self.crop_size = params['crop_size']
        self.clean_targs = params['clean_targs']
        self.lr = params['lr']
        self.epochs = params['epochs']
        self.bs = params['bs']

        self.train_dl, self.val_dl = self.__getdataset__()
        self.optimizer = self.__getoptimizer__()
        self.scheduler = self.__getscheduler__()
        self.loss_fn = self.__getlossfn__(params['lossfn'])

    
    def train(self):
        
        for _ in range(self.epochs):
            tr_loss = 0
            self.architecture.train(True)
            for _, (source, target) in tqdm(enumerate(self.train_dl)):
                source = source.cuda()
                target = target.cuda()
                _op = self.architecture(Variable(source))
                _loss = self.loss_fn(_op, Variable(target))
                tr_loss += _loss

                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
            
            val_loss = self.evaluate()
            self.scheduler.step(val_loss)
            print(f'Training loss = {tr_loss}, Validation loss = {val_loss}')


    def evaluate(self):
        
        val_loss = 0
        self.architecture.train(False)

        for _, (source, target) in enumerate(self.val_dl):
            source = source.cuda()
            target = target.cuda()
            _op = self.architecture(Variable(source))
            _loss = self.loss_fn(_op, Variable(target))
            val_loss += _loss
        
        return val_loss

    
    def __getdataset__(self):
        
        train_ds = NoisyDataset(self.train_dir, crop_size=self.crop_size, train_noise_model=self.noise_model,
                                        clean_targ=self.clean_targs)
        train_dl = DataLoader(train_ds, batch_size=self.bs, shuffle=True)

        val_ds = NoisyDataset(self.val_dir, crop_size=self.crop_size, train_noise_model=self.noise_model,
                                        clean_targ=True)
        val_dl = DataLoader(val_ds, batch_size=self.bs)

        return train_dl, val_dl

    def __getoptimizer__(self):
        
        return optim.Adam(self.architecture.parameters(), self.lr)

    def __getscheduler__(self):
        
        return lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=self.epochs/4, factor=0.5, verbose=True)

    def __getlossfn__(self, lossfn):
        
        if lossfn == 'l2':
            return nn.MSELoss()
        elif lossfn == 'l1':
            return nn.L1Loss()
        else:
            raise ValueError('No such loss function supported')

    

