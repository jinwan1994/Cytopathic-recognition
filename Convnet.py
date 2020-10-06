# -*- coding: utf-8 -*-

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms

import time
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image

# use PIL Image to read image
def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label

####dataset loading..... #****val_set==test_set
    
data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(60),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        'test': transforms.Compose([
            transforms.Resize(60),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
    }

use_gpu = torch.cuda.is_available()

batch_size = 50
num_class = 2

# image_datasets={}

train_data = customData(img_path='D:/BaiduNetdiskDownload/image_classification_with_pytorch-master/image_classification_with_pytorch-master/your_data_folder_names/',
                                         txt_path=('D:/BaiduNetdiskDownload/image_classification_with_pytorch-master/image_classification_with_pytorch-master/your_data_folder_names/TxtFile/train.txt'),
                                         data_transforms=data_transforms,
                                         dataset='train',
                                         )

test_data = customData(img_path='D:/BaiduNetdiskDownload/image_classification_with_pytorch-master/image_classification_with_pytorch-master/your_data_folder_names/',
                                       txt_path=('D:/BaiduNetdiskDownload/image_classification_with_pytorch-master/image_classification_with_pytorch-master/your_data_folder_names/TxtFile/test.txt'),
                                       data_transforms=data_transforms,
                                       dataset='test',
                                       )
print('num of train_set',len(train_data))
print('num of test_set',len(test_data))


train_loader = DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=True)

###定义卷积神经网络--钟亚
class CON(nn.Module):
    def __init__(self):
        super(CON, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 60, 60)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=10,            # n_filters
                kernel_size=11,              # filter size
                stride=1,                   # filter movement/step
            ),                              # output shape (10, 50, 50)
            nn.ReLU(),                      # activation
            nn.AvgPool2d(kernel_size=2,stride=2),    # choose max value in 2x2 area, output shape (6,28,28)
        )
        self.conv2 = nn.Sequential(         # input shape (10, 25, 25)
            nn.Conv2d(10, 7, 6, 1),     # output shape (7, 20, 20)
            nn.BatchNorm2d(7),
            nn.ReLU(),                      # activation
            nn.AvgPool2d(2,2),                # output shape (7, 10, 10)

        )

        self.out = nn.Sequential(
            nn.Linear(7 * 10 * 10, 2),   # fully connected layer, output 02 classes
            nn.Tanh(),
            )

    def forward(self, x):

        x = self.conv1(x)

        x = self.conv2(x)

        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)

        output = self.out(x)

        return output 


# Net = CNN()

Net = CON()

####define optimizer and loss function
optimizer = optim.SGD(Net.parameters(), lr=0.0001, momentum=0.9)          
loss_func = nn.CrossEntropyLoss()

since = time.time()
####training 
for epoch in range(30):#设置训练的迭代次数
    
    running_loss = 0.0
    running_corrects = 0.0
    best_acc = 0.0

    for data in train_loader:
    
        imgs, labels = data

        inputs, labels = Variable(imgs), Variable(labels)

        optimizer.zero_grad()

        outputs = Net(imgs)
        
        _, preds = torch.max(outputs.data, 1)

        loss = loss_func(outputs, labels)  

        loss.backward()

        optimizer.step()#利用计算的得到的梯度对参数进行更新
        running_loss += loss.item()
        running_corrects += torch.sum(preds == labels.data).to(torch.float32)
    #打印log信息
        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects / len(train_data)
        
    epoch_loss = running_loss / len(train_data)
    epoch_acc = running_corrects / len(train_data)
    print(' Loss: ',epoch_loss,' Acc: ' ,epoch_acc)

    if  epoch_acc > best_acc:
        best_acc = epoch_acc
               

time_elapsed = time.time() - since
print('\nTraining complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best Acc: {:4f}'.format(best_acc))
print('\nFinished Training')

####testing 
for data1 in test_loader:
        
    imgs_test,labels_test = data1
        
    test_output= Net(imgs_test)
        
    pred_y = torch.max(test_output, 1)[1].data.numpy()
    
print( 'prediction number:',pred_y[0:10])
    
print('real number:',labels_test.numpy()[0:10] )









