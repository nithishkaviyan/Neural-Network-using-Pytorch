
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


# In[2]:


##CIFAR10 data
CIFAR_data=h5py.File("CIFAR10.hdf5",'r')
lis=list(CIFAR_data)
x_train=np.float32(np.array(CIFAR_data[lis[1]]))
y_train=np.int32(np.array(CIFAR_data[lis[3]]))
x_test=np.float32(np.array(CIFAR_data[lis[0]]))
y_test=np.int32(np.array(CIFAR_data[lis[2]]))
CIFAR_data.close()


# In[ ]:


##Neural Network Model
class CIFARmodel(nn.Module):
    def __init__(self):
        super(CIFARmodel,self).__init__()
        self.l1=nn.Conv2d(3,64,4,stride=1,padding=2)
        self.l1_bn=nn.BatchNorm2d(64)
        self.l2=nn.Conv2d(64,64,4,stride=1,padding=2)
        self.l3_do=nn.Dropout2d()
        self.l3=nn.Conv2d(64,64,4,stride=1,padding=2)
        self.l3_bn=nn.BatchNorm2d(64)
        self.l4=nn.Conv2d(64,64,4,stride=1,padding=2)
        self.l5_do=nn.Dropout2d()
        self.l5=nn.Conv2d(64,64,4,stride=1,padding=2)
        self.l5_bn=nn.BatchNorm2d(64)
        self.l6=nn.Conv2d(64,64,3)
        self.l7_do=nn.Dropout2d()
        self.l7=nn.Conv2d(64,64,3)
        self.l7_bn=nn.BatchNorm2d(64)
        self.l8=nn.Conv2d(64,64,3)
        self.l8_bn=nn.BatchNorm2d(64)
        self.l9_do=nn.Dropout2d()
        self.l9=nn.Linear(1024,500)
        self.l10=nn.Linear(500,500)
        self.l11=nn.Linear(500,10)
        
    def forward(self,x):
        x=F.relu(self.l1_bn(self.l1(x)))
        x=F.relu(F.max_pool2d(self.l2(x),2,stride=2))
        x=F.relu(self.l3_bn(self.l3(self.l3_do(x))))
        x=F.relu(F.max_pool2d(self.l4(x),2,stride=2))
        x=F.relu(self.l5_bn(self.l5(self.l5_do(x))))
        x=F.relu(self.l6(x))
        x=F.relu(self.l7_bn(self.l7(self.l7_do(x))))
        x=F.relu(self.l8_bn(self.l8(x)))
        x=self.l9_do(x)
        x=x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])
        x=F.relu(self.l9(x))
        x=F.relu(self.l10(x))
        x=self.l11(x)
        return F.log_softmax(x, dim=1)
        


# In[ ]:


##Define the model
model=CIFARmodel()
model.cuda()

##AdamOptimizer
optimizer=optim.Adam(model.parameters(),lr=0.001)

batch_size=100
num_epochs=100
L=len(x_train)
train_loss=[]

for epoch in range(num_epochs):
    ##Shuffle the dataset
    order=np.random.permutation(L)
    x_train=x_train[order,:,:,:]
    y_train=y_train[order]
    train_acc=[]
    
    if(epoch>12):
        for group in optimizer.param_groups:
            for p in group['params']:
                state=optimizer.state[p]
                if (state['step']>=1024):
                    state['step']=1000
    
    for i in range(0,L,batch_size):
        x_tr=torch.FloatTensor(x_train[i:i+batch_size,:])
        y_tr=torch.LongTensor(y_train[i:i+batch_size])
        
        dat,tar=Variable(x_tr).cuda(),Variable(y_tr).cuda()
        
        ##Set gradients to zero
        optimizer.zero_grad()
        
        ##Forward Propagation
        forw_out=model(dat)
        
        ##ObjectiveFunction
        loss=F.nll_loss(forw_out,tar)
        
        ##Backward propagation
        loss.backward()
        train_loss.append(loss.data[0])
        
        ##Update parameters
        optimizer.step()
        
        ##Training Accuracy calculation
        predict=forw_out.data.max(1)[1]
        accuracy=(float(predict.eq(tar.data).sum())/float(batch_size))*100.0
        train_acc.append(accuracy)
    
    ##Epoch accuracy
    epoch_acc=np.mean(train_acc)
    print("Epoch: "+str(epoch+1)+" Train Accuracy: "+str(epoch_acc))


# In[ ]:


##Test accuracy
model.eval()
test_acc=[]

for j in range(0,len(x_test),batch_size): 
    x_tes=torch.FloatTensor(x_test[j:j+batch_size,:])
    y_tes=torch.LongTensor(y_test[j:j+batch_size])
    dat_tes,tar_tes=Variable(x_tes).cuda(),Variable(y_tes).cuda()
    optimizer.zero_grad()
    output_tes=model(dat_tes)
    test_loss=F.nll_loss(output_tes,tar_tes)
    pred_tes=output_tes.data.max(1)[1]
    acc_tes=(float(pred_tes.eq(tar_tes.data).sum())/float(batch_size))*100.0
    test_acc.append(acc_tes)
    
print("Test Accuracy: "+str(np.mean(test_acc)))

