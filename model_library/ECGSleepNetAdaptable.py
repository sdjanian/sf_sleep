import torch as th
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


       
     
# ChatGPT              
class ResBlock(nn.Module):
    def __init__(self, Lin, Lout, filter_len, dropout, subsampling, momentum, maxpool_padding=0):
        assert filter_len % 2 == 1
        super(ResBlock, self).__init__()
        self.Lin = Lin
        self.Lout = Lout
        self.filter_len = filter_len
        self.dropout = dropout
        self.subsampling = subsampling
        self.momentum = momentum
        self.maxpool_padding = maxpool_padding

        self.bn1 = nn.BatchNorm1d(self.Lin, momentum=self.momentum, affine=True)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(self.dropout)
        self.conv1 = nn.Conv1d(self.Lin, self.Lin, self.filter_len, stride=self.subsampling, padding=self.filter_len // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(self.Lin, momentum=self.momentum, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.conv2 = nn.Conv1d(self.Lin, self.Lout, self.filter_len, stride=1, padding=self.filter_len // 2, bias=False)

        if self.Lin == self.Lout and self.subsampling > 1:
            self.maxpool = nn.MaxPool1d(self.subsampling, padding=self.maxpool_padding)

    def forward(self, x):
        res = x  # Save the input for residual connection
        
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv2(x)

        # If the input and output channels are the same
        if self.Lin == self.Lout:
            if self.subsampling > 1:
                res = self.maxpool(res)

            # Ensure the sizes match before adding
            if x.size(2) != res.size(2):
                res = F.interpolate(res, size=x.size(2))

            x = x + res  # Add the residual
        return x

        

class ECGSleepNetAdaptable(nn.Module):

    def __init__(self, to_combine=False,nb_classes = 5,n_timestep = 54000):#, filter_len):
        super(ECGSleepNetAdaptable, self).__init__()
        self.filter_len = 17#33
        self.filter_num = 64#16
        self.padding = self.filter_len//2
        self.dropout = 0.5
        self.momentum = 0.1
        self.subsampling = 4
        self.n_channel = 1
        self.n_timestep = n_timestep#54000#//2
        #self.n_output = 5
        self.n_output = nb_classes
        self.to_combine = to_combine
        
        # input convolutional block
        # 1 x 54000
        self.conv1 = nn.Conv1d(1, self.filter_num, self.filter_len, stride=1, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)
        self.relu1 = nn.ReLU()
        
        # 64 x 54000
        self.conv2_1 = nn.Conv1d(self.filter_num, self.filter_num, self.filter_len, stride=self.subsampling, padding=self.padding, bias=False)
        self.bn2 = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(self.dropout)
        self.conv2_2 = nn.Conv1d(self.filter_num, self.filter_num, self.filter_len, stride=1, padding=self.padding, bias=False)
        self.maxpool2 = nn.MaxPool1d(self.subsampling)
        #self.bn_input = nn.BatchNorm1d(self.filter_num, momentum=self.momentum, affine=True)

        # 64 x 13500
        self.resblock1 = ResBlock(self.filter_num, self.filter_num, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock2 = ResBlock(self.filter_num, self.filter_num, self.filter_len,
                self.dropout, self.subsampling, self.momentum)
        self.resblock3 = ResBlock(self.filter_num, self.filter_num*2, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock4 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
                
        # 128 x 844
        self.resblock5 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock6 = ResBlock(self.filter_num*2, self.filter_num*2, self.filter_len,
                self.dropout, self.subsampling, self.momentum)
        self.resblock7 = ResBlock(self.filter_num*2, self.filter_num*3, self.filter_len,
                self.dropout, 1, self.momentum)                
        self.resblock8 = ResBlock(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=1)
                
        # 192 x 53
        self.resblock9 = ResBlock(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock10 = ResBlock(self.filter_num*3, self.filter_num*3, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2)
        self.resblock11 = ResBlock(self.filter_num*3, self.filter_num*4, self.filter_len,
                self.dropout, 1, self.momentum)
        self.resblock12 = ResBlock(self.filter_num*4, self.filter_num*4, self.filter_len,
                self.dropout, self.subsampling, self.momentum, maxpool_padding=2)
                
        # 256 x 4
        self.resblock13 = ResBlock(self.filter_num*4, self.filter_num*5, self.filter_len,
                self.dropout, 1, self.momentum)

        # 320 x 4
        self.bn_output = nn.BatchNorm1d(self.filter_num*5, momentum=self.momentum, affine=True)
        self.relu_output = nn.ReLU()
        
        #if not self.to_combine:
        #dummy = self._forward(Variable(th.ones(1,self.n_channel, self.n_timestep)))
        #self.fc_output = nn.Linear(dummy.size(1), self.n_output)
        
        # Dynamically calculate the size of the output after convolutions
        dummy = self._forward(Variable(th.ones(1, self.n_channel, self.n_timestep)))
        flattened_size = dummy.size(1)
        
        # Adjust the fully connected layer based on the flattened size
        self.fc_output = nn.LazyLinear(self.n_output)        

    def _forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        res = x
        x = self.conv2_1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv2_2(x)
        x = x+self.maxpool2(res)

        #x = self.bn_input(x)
        x = self.resblock1(x)
        #print(f"Output after resblock1: {x.size()}")
        x = self.resblock2(x)
        #print(f"Output after resblock2: {x.size()}")
        x = self.resblock3(x)
        #print(f"Output after resblock3: {x.size()}")
        x = self.resblock4(x)
        #print(f"Output after resblock4: {x.size()}")
        x = self.resblock5(x)
        #print(f"Output after resblock5: {x.size()}")
        x = self.resblock6(x)
        #print(f"Output after resblock6: {x.size()}")
        x = self.resblock7(x)
        #print(f"Output after resblock7: {x.size()}")
        x = self.resblock8(x)
        #print(f"Output after resblock8: {x.size()}")
        if hasattr(self, 'to_combine') and self.to_combine:
            return x
        x = self.resblock9(x)
        #print(f"Output after resblock9: {x.size()}")
        x = self.resblock10(x)
        #print(f"Output after resblock10: {x.size()}")
        x = self.resblock11(x)
        #print(f"Output after resblock11: {x.size()}")
        x = self.resblock12(x)
        #print(f"Output after resblock12: {x.size()}")
        x = self.resblock13(x)
        #print(f"Output after resblock12: {x.size()}")

        
        x = self.bn_output(x)
        x = self.relu_output(x)
        #print(f"Output after output: {x.size()}")


        x = x.view(x.size(0), -1)
        #print(f"Output after final X: {x.size()}")

        return x

    def forward(self, x):
        h = self._forward(x)
        if not hasattr(self, 'to_combine') or not self.to_combine:
            x = self.fc_output(h)
        
        return x, h
        
    def load_param(self, model_path): 
        model = th.load(model_path)
        if type(model)==nn.DataParallel and hasattr(model, 'module'):
            model = model.module
        if hasattr(model, 'state_dict'):
            model = model.state_dict()
        self.load_state_dict(model)
        
    def fix_param(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def unfix_param(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def init(self, method='orth'):
        pass
if __name__ == '__main__':
    

  
    
    ECGSleepNet64 = ECGSleepNetAdaptable(nb_classes = 5,n_timestep = 17280)
    ECGSleepNet200 = ECGSleepNetAdaptable(nb_classes = 5,n_timestep = 200*270)


    import torchsummary as ts

    ts.summary(ECGSleepNet64, (1,17280))
    ts.summary(ECGSleepNet200, (1,54000))

    


    #model.eval()

    
