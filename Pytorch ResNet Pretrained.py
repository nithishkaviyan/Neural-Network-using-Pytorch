
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

##Load CIFAR 100 dataset
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor()
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

train_data = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
test_data = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

##Load pre-trained model
model=models.resnet18(pretrained=True)

model_urls='https://download.pytorch.org/models/resnet18-5c106cde.pth'
model.load_state_dict(model_zoo.load_url(model_urls,model_dir='./'))

model.fc=nn.Linear(model.fc.in_features,100)



##Define the model
model.cuda()

##Loss
loss_fn=nn.CrossEntropyLoss()

##Optimizer
optimizer=optim.Adam(model.parameters(),lr=0.001)
#optimizer=optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

num_epochs=10
batch_size=128
train_loss=[]
#scheduler=optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,20,30,40],gamma=0.5)

for epoch in range(num_epochs):
    train_acc=[]
    epoch_loss=[]
    #scheduler.step()
    
    for data in train_data:
        x_tr,y_tr=data
        dat,tar=Variable(x_tr).cuda(),Variable(y_tr).cuda()
        #dat,tar=x_tr,y_tr
        
        ##Set grad to zero
        optimizer.zero_grad()
    
        ##Forward prop
        dat=F.interpolate(dat,size=[224,224])
        forw_out=model(dat)
    
        ##Compute loss
        loss=loss_fn(forw_out,tar)
    
        ##Back propagation
        loss.backward()
        train_loss.append(loss.data[0])
        epoch_loss.append(loss.data[0])
        
        ##Update Step
        optimizer.step()
    
        ##Training accuracy
        predict=forw_out.data.max(1)[1]
        accuracy=(float(predict.eq(tar.data).sum())/float(batch_size))*100.0
        train_acc.append(accuracy)
        
    ##Epoch accuracy
    epoch_acc=np.mean(train_acc)
    print("Epoch: "+str(epoch+1)+" Loss: "+str(np.mean(epoch_loss))+" Train accuracy: "+str(epoch_acc))
                  
    torch.save(model,'mod_save.ckpt')
     
    ##Test accuracy
    mod=torch.load('mod_save.ckpt')
    test_acc=[]

    for data in test_data:
        x_tes,y_tes=data
        dat_tes,tar_tes=Variable(x_tes).cuda(),Variable(y_tes).cuda()
        optimizer.zero_grad()
        dat_tes=F.interpolate(dat_tes,size=[224,224])
        output_tes=mod(dat_tes)
        test_loss=F.nll_loss(output_tes,tar_tes)
        pred_tes=output_tes.data.max(1)[1]
        acc_tes=(float(pred_tes.eq(tar_tes.data).sum())/float(batch_size))*100.0
        test_acc.append(acc_tes)  
            
    print("Test Accuracy: "+str(np.mean(test_acc)))

