import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np

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

##Basic Block
class BasicBlock(nn.Module):
  def __init__(self,np,nc,s):
    super(BasicBlock,self).__init__()
    self.b1=nn.Conv2d(np,nc,3,stride=s,padding=1)
    self.b1_bn=nn.BatchNorm2d(nc)
    self.b2=nn.Conv2d(nc,nc,3,stride=s,padding=1)
    self.b2_bn=nn.BatchNorm2d(nc)
    self.downsample=nn.Sequential()
    if(s!=1):
      self.downsample=nn.Sequential(nn.Conv2d(np,nc,1,stride=2),
                                    nn.BatchNorm2d(nc))
  def forward(self,x):
    res=x
    x=F.relu(self.b1_bn(self.b1(x)))
    x=self.b2_bn(self.b2(x))
    #res=F.interpolate(self.downsample(res),size=[x.shape[2],x.shape[3]])
    res=self.downsample(res)
    x=F.interpolate(x,size=[res.shape[2],res.shape[3]])
    x+=res
    return x

##ResNet
class ResNet(nn.Module):
    def __init__(self,block):
        super(ResNet, self).__init__()
        self.l1=nn.Conv2d(3,32,3,stride=1,padding=1)
        self.l1_bn=nn.BatchNorm2d(32)
        self.l1_do=nn.Dropout2d(0.2)
        self.l2=block(32,32,1)
        self.l21=block(32,32,1)
        self.l3=block(32,64,2)
        self.l31=block(64,64,1)
        self.l32=block(64,64,1)
        self.l33=block(64,64,1)
        self.l4=block(64,128,2)
        self.l41=block(128,128,1)
        self.l42=block(128,128,1)
        self.l43=block(128,128,1)
        self.l5=block(128,256,2)
        self.l51=block(256,256,1)
        self.l6=nn.Linear(1024,100)
    
    def forward(self,x):
        x=self.l1_do(F.relu(self.l1_bn(self.l1(x))))
        x=self.l2(x)
        x=self.l21(x)
        x=self.l3(x)
        x=self.l31(x)
        x=self.l32(x)
        x=self.l33(x)
        x=self.l4(x)
        x=self.l41(x)
        x=self.l42(x)
        x=self.l43(x)
        x=self.l5(x)
        x=F.max_pool2d(self.l51(x),3,stride=2,padding=1)
        x=x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x=self.l6(x)
        return F.log_softmax(x, dim=1)

##Define the model
model=ResNet(BasicBlock)
model.cuda()

##Loss
loss_fn=nn.CrossEntropyLoss()

##Optimizer
optimizer=optim.Adam(model.parameters(),lr=0.001)
#optimizer=optim.SGD(model.parameters(),lr=0.0001,momentum=0.9)

num_epochs=40
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
    
    ##Save the model
    if((epoch+1)%5==0):
      torch.save(model,'save_model.ckpt')
 

    if((epoch+1)%10==0):
      ##Test accuracy
      model.eval()
      test_acc=[]

      for data in test_data:
        x_tes,y_tes=data
        dat_tes,tar_tes=Variable(x_tes).cuda(),Variable(y_tes).cuda()
        optimizer.zero_grad()
        output_tes=model(dat_tes)
        test_loss=F.nll_loss(output_tes,tar_tes)
        pred_tes=output_tes.data.max(1)[1]
        acc_tes=(float(pred_tes.eq(tar_tes.data).sum())/float(batch_size))*100.0
        test_acc.append(acc_tes)
    
      print("Test Accuracy: "+str(np.mean(test_acc)))

##Test accuracy
model.eval()
test_acc=[]

for data in test_data:
    x_tes,y_tes=data
    dat_tes,tar_tes=Variable(x_tes).cuda(),Variable(y_tes).cuda()
    optimizer.zero_grad()
    output_tes=model(dat_tes)
    test_loss=F.nll_loss(output_tes,tar_tes)
    pred_tes=output_tes.data.max(1)[1]
    acc_tes=(float(pred_tes.eq(tar_tes.data).sum())/float(batch_size))*100.0
    test_acc.append(acc_tes)
    
print("Test Accuracy: "+str(np.mean(test_acc)))









