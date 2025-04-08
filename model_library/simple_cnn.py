import torch
"""
class Simple1DCNN(torch.nn.Module):
    def __init__(self, input_size=1920, num_of_sleep_stages=5):
        super(Simple1DCNN, self).__init__()
        self.cnn1 = torch.nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.act1 = torch.nn.ReLU()
        self.max1 = torch.nn.MaxPool1d(2)
        self.cnn2 = torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.act2 = torch.nn.ReLU()
        self.max2 = torch.nn.MaxPool1d(2)        
        self.cnn3 = torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.act3 = torch.nn.ReLU()
        self.max3 = torch.nn.MaxPool1d(2)          
        self.cnn4 = torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)
        self.bn4 = torch.nn.BatchNorm1d(256)
        self.act4 = torch.nn.ReLU()
        self.max4 = torch.nn.MaxPool1d(2)        
        self.cnn5 = torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=5, stride=1)
        self.bn5 = torch.nn.BatchNorm1d(512)
        self.act5 = torch.nn.ReLU()
        self.max5 = torch.nn.MaxPool1d(2)            
        self.dense1 = torch.nn.LazyLinear(out_features = 4096)
        #self.dense1 = torch.nn.Linear(in_features=30656, out_features=30656)
        self.bnd1 = torch.nn.BatchNorm1d(4096)
        self.actd1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(in_features=4096, out_features=64)
        self.actd2 = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(in_features=64, out_features=num_of_sleep_stages)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.max1(x)
        x = self.cnn2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.max2(x)
        x = self.cnn3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.max3(x)
        x = self.cnn4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.max4(x)        
        x = self.cnn5(x)
        x = self.bn5(x)
        x = self.act5(x)
        x = self.max5(x)          
        x = x.view(x.size(0), -1)#torch.flatten(x, start_dim=1)  # Flatten the tensor for the dense layer
        x = self.dense1(x)
        x = self.bnd1(x)
        x = self.actd1(x)
        x = self.dense2(x)
        x = self.actd2(x)
        x = self.output_layer(x)
        return x
 """   
class Simple1DCNN(torch.nn.Module):
    def __init__(self, input_size=1920, num_of_sleep_stages=5,num_layers=1,kernel_size:list=[5]):
        super(Simple1DCNN, self).__init__()
        self.num_layers = num_layers
        self.conv_bn_relu_layers = torch.nn.ModuleList()
        self.kernel_size = kernel_size
        
        input_channels = 1
        output_channels = 32 # Filter size
        # Create 1D convolutional, BatchNorm, and ReLU layers
        for _ in range(num_layers):
            self.conv_bn_relu_layers.append(torch.nn.Sequential(
                torch.nn.Conv1d(input_channels, output_channels, kernel_size=self.kernel_size[_]),
                torch.nn.BatchNorm1d(output_channels),
                torch.nn.ReLU(),
                torch.nn.MaxPool1d(2)
            ))
            input_channels = output_channels  # Update input_channels for the next layer
            output_channels = output_channels * 2
            
        self.dense1 = torch.nn.LazyLinear(out_features = 4096)
        #self.dense1 = torch.nn.Linear(in_features=30656, out_features=30656)
        self.bnd1 = torch.nn.BatchNorm1d(4096)
        self.actd1 = torch.nn.ReLU()
        self.dense2 = torch.nn.Linear(in_features=4096, out_features=64)
        self.actd2 = torch.nn.ReLU()
        self.output_layer = torch.nn.Linear(in_features=64, out_features=num_of_sleep_stages)

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv_bn_relu_layers[i](x)      
        x = x.view(x.size(0), -1)#torch.flatten(x, start_dim=1)  # Flatten the tensor for the dense layer
        x = self.dense1(x)
        x = self.bnd1(x)
        x = self.actd1(x)
        x = self.dense2(x)
        x = self.actd2(x)
        x = self.output_layer(x)
        return x    
    

model = Simple1DCNN(num_layers=3,kernel_size=[5,3,5])
#t = torch.rand((1,1,1920))
#print(model(torch.tensor(t)).size)
"""
class FCN(nn.Module):
    def __init__(self, input_shape, nb_classes, kernel_size=8):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=128, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=kernel_size*2//3, padding=kernel_size*2//6)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=kernel_size*4//9, padding=kernel_size*4//18)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, nb_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
"""