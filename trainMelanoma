import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import pandas as pd
import numpy as np
import gc
import os
import cv2
import time
import datetime
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from efficientnet_pytorch import EfficientNet
# %matplotlib inline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, imfolder: str, train: bool = True, transforms = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            train (bool): flag of whether a training dataset is being initialized or testing one
            transforms: image transformation method to be applied
            
        """
        self.df = df
        self.imfolder = imfolder
        self.transforms = transforms
        self.train = train
        
    def __getitem__(self, index):
        im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + '.jpg')
        x = cv2.imread(im_path)

        if self.transforms:
            x = self.transforms(x)
            
        if self.train:
            y = self.df.loc[index]['target']
            return x, y, im_path
        else:
            return x
    
    def __len__(self):
        return len(self.df)
    
    
class Net(nn.Module):
    def __init__(self, arch):
        super(Net, self).__init__()
        self.arch = arch
        if 'ResNet' in str(arch.__class__):
            self.arch.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        if 'EfficientNet' in str(arch.__class__):
            self.arch._fc = nn.Linear(in_features=1280, out_features=1, bias=True)
        
    def forward(self, x):
        """
        No sigmoid in forward because we are going to use BCEWithLogitsLoss
        Which applies sigmoid for us when calculating a loss
        """
        x = self.arch(x)
        return x

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=32. / 255.,saturation=0.5),
    transforms.Cutout(scale=(0.05, 0.007), value=(0, 0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

arch = EfficientNet.from_pretrained('efficientnet-b0')  # Going to use efficientnet-b0 NN architecture
# skf = StratifiedKFold(n_splits=3, random_state=999, shuffle=True)
#skf = GroupKFold(n_splits=5)

train_df = pd.read_csv('dataset/train_sep.csv')
validation_df = pd.read_csv('dataset/validation.csv')
test_df = pd.read_csv('dataset/test.csv')
test = MelanomaDataset(df=test_df,
                       imfolder='dataset/test/test/', 
                       train=False,
                       transforms=train_transform)

train_df['patient_id'] = train_df['patient_id'].fillna(0)
validation_df['patient_id'] = train_df['patient_id'].fillna(0)

# At least fixing some random seeds. 
# It is still impossible to make results 100% reproducible when using GPU
warnings.simplefilter('ignore')
torch.manual_seed(47)
np.random.seed(47)


from torch.utils.data import WeightedRandomSampler
epochs = 1  # Number of epochs to run
model_path = 'model.pth'  # Path and filename to save model to
es_patience = 3  # Early Stopping patience - for how many epochs with no improvements to wait
TTA = 3 # Test Time Augmentation rounds
#oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions
preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)  # Predictions for test test

# We stratify by target value, thus, according to sklearn StratifiedKFold documentation
# We can fill `X` with zeroes of corresponding length to use it as a placeholder
# since we only need `y` to stratify the data
# for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target']), 1):

# for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target'], groups=train_df['patient_id'].tolist()), 1):
#     print('=' * 20, 'Fold', fold, '=' * 20)

best_val = None  # Best validation score within this fold
patience = es_patience  # Current patience counter
arch = EfficientNet.from_pretrained('efficientnet-b0')
model = Net(arch=arch)  # New model for each fold
model = model.to(device)
  
    
optim = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer=optim, mode='max', patience=1, verbose=True, factor=0.2)
criterion = nn.BCEWithLogitsLoss()
labels_unique, counts_train = np.unique(train_df['target'], return_counts=True)
labels_unique, counts_validation = np.unique(validation_df['target'], return_counts=True)
   
train = MelanomaDataset(df=train_df.reset_index(drop=True), 
                        imfolder='dataset/train/', 
                        train=True, 
                        transforms=train_transform)
val = MelanomaDataset(df=validation_df.reset_index(drop=True), 
                        imfolder='dataset/validation/', 
                        train=True, 
                        transforms=test_transform)
print(len(train))
# Train weight balancing
class_weights_train = [sum(counts_train)/c for c in counts_train]
train_weights = [class_weights_train[e] for e in train_df['target']]
sampler = WeightedRandomSampler(train_weights, len(train_df))
    
# Validation weight balancing
class_weights_validation = [sum(counts_validation)/c for c in counts_validation]
validation_weights = [class_weights_validation[e] for e in validation_df['target']]
sampler_val = WeightedRandomSampler(validation_weights, len(validation_df))
  
train_loader = DataLoader(dataset=train, sampler=sampler, batch_size=64, num_workers=2)
val_loader = DataLoader(dataset=val, sampler=sampler_val, batch_size=16, num_workers=2)
test_loader = DataLoader(dataset=test, batch_size=16, shuffle=False, num_workers=2)
    
for epoch in range(epochs):
    start_time = time.time()
    correct = 0
    epoch_loss = 0
    model.train()
    print(len(train_loader))
    for x, y, img in train_loader:
        #print(img)
        x = torch.tensor(x, device=device, dtype=torch.float32)
        y = torch.tensor(y, device=device, dtype=torch.float32)
        optim.zero_grad()
        z = model(x)
        loss = criterion(z, y.unsqueeze(1))
        loss.backward()
        optim.step()
        pred = torch.round(torch.sigmoid(z))  # round off sigmoid to obtain predictions
        correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
        epoch_loss += loss.item()
    train_acc = correct / len(train_df)

    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((7548, 1), dtype=torch.float32, device=device)
    with torch.no_grad():  # Do not calculate gradient since we are only predicting
       # Predicting on validation set
        for j, (x_val, y_val, img) in enumerate(val_loader):
            #print(img)
            x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
            y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
            z_val = model(x_val)
            val_pred = torch.sigmoid(z_val)
            val_preds[j*x_val.shape[0]:j*x_val.shape[0] + x_val.shape[0]] = val_pred
        val_acc = accuracy_score(validation_df['target'].values, torch.round(val_preds.cpu()))
        val_roc = roc_auc_score(validation_df['target'].values, val_preds.cpu())
            
        print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.3f} | Training time: {}'.format(
        epoch + 1, 
        epoch_loss, 
        train_acc, 
        val_acc, 
        val_roc, 
        str(datetime.timedelta(seconds=time.time() - start_time))[:7]))
            
        scheduler.step(val_roc)
        # During the first iteration (first epoch) best validation is set to None
        if not best_val:
            best_val = val_roc  # So any validation roc_auc we have is the best one for now
            torch.save(model, model_path)  # Saving the model
            continue
              
        if val_roc >= best_val:
            best_val = val_roc
            patience = es_patience  # Resetting patience since we have new best validation accuracy
            torch.save(model, model_path)  # Saving current best model
        else:
         
            patience -= 1
            if patience == 0:
                print('Early stopping. Best Val roc_auc: {:.3f}'.format(best_val))
                break
                
model = torch.load(model_path)  # Loading best model of this fold
model.eval()  # switch model to the evaluation mode
val_preds = torch.zeros((7548, 1), dtype=torch.float32, device=device)
with torch.no_grad():
    # Predicting on validation set once again to obtain data for OOF
    for j, (x_val, y_val, img) in enumerate(val_loader):
        x_val = torch.tensor(x_val, device=device, dtype=torch.float32)
        y_val = torch.tensor(y_val, device=device, dtype=torch.float32)
        z_val = model(x_val)
        val_pred = torch.sigmoid(z_val)
        val_preds[j*x_val.shape[0]:j*x_val.shape[0] + x_val.shape[0]] = val_pred
    #oof[7548] = val_preds.cpu().numpy()
        
    # Predicting on test set
    for _ in range(TTA):
       for i, x_test in enumerate(test_loader):
            x_test = torch.tensor(x_test, device=device, dtype=torch.float32)
            z_test = model(x_test)
            z_test = torch.sigmoid(z_test)
            preds[i*x_test.shape[0]:i*x_test.shape[0] + x_test.shape[0]] += z_test
    preds /= TTA
        
del train, val, train_loader, val_loader, x, y, x_val, y_val, img
gc.collect()
sub = pd.read_csv('dataset/sample_submission.csv')
sub['target'] = (preds.cpu().numpy().reshape(-1,).round())
sub.to_csv('submission.csv', index=False)

