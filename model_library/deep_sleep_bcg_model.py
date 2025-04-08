import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, num_filters, kernel_size, stride):
        super(ResidualBlock, self).__init__()
        self.padding = kernel_size // 2  # Ensures output size is consistent
        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size, stride, padding=self.padding)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, stride, padding=self.padding)
        self.relu = nn.ReLU()
    

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class DeepSleep(nn.Module):
    def __init__(self, num_filters=64, kernel_size=25, stride=1, num_residual_blocks = 8, nb_classes = 4):
        super(DeepSleep, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, 
                               kernel_size=kernel_size, stride=stride)
        #self.batchnorm_initial = nn.BatchNorm1d(num_filters)  # Batch normalization layer

        self.maxpool = nn.MaxPool1d(2)
        self.resblocks = nn.Sequential(*[ResidualBlock(num_filters, kernel_size, stride) 
                                         for _ in range(num_residual_blocks)])
        self.maxpool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=128, num_layers=3, 
                            batch_first=True, bidirectional=True)
        self.dense = nn.Linear(64, 256)  # Adjust the number of features to 256
        #self.dense = nn.LazyLinear(512)
        self.batchnorm = nn.BatchNorm1d(256)  # Batch normalization layer
        self.classifier = nn.LazyLinear(nb_classes)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        print("After conv1:", out.shape)
        
        out = self.maxpool(out)
        print("After maxpool:", out.shape)
        
        out = self.resblocks(out)
        print("After resblocks:", out.shape)
        
        out = self.maxpool2(out)
        print("After maxpool2:", out.shape)
        
        res = self.adaptive_pool(out).squeeze(-1)  # Save maxpool output
        print("After adaptive_pool:", res.shape)
        
        out = out.transpose(1, 2)  # To match the LSTM's expected input
        out, _ = self.lstm(out)
        print("After LSTM:", out.shape)
        
        out = out[:, -1, :]  # Take the output from the final time step
        res = self.dense(res)  # adjust the residual to have the same number of features
        res = self.batchnorm(res)  # Apply batch normalization
        
        print("Before residual addition - out:", out.shape, ", res:", res.shape)
        
        out = out + res  # Add residual connection
        out = self.relu(out)
        out = self.classifier(out)
        return out


class ResidualBlock2(nn.Module):
    def __init__(self, num_filters, kernel_size, stride=1):
        super(ResidualBlock2, self).__init__()
        #self.padding = kernel_size // 2  # Ensures output size is consistent
        #self.padding = (3 * 925 - 4 + kernel_size) // 2
        self.padding = kernel_size // 2  # Ensures output size is consistent

        self.conv1 = nn.Conv1d(num_filters, num_filters, kernel_size, stride, padding=self.padding)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size, stride, padding=self.padding)
        self.relu = nn.ReLU()
    

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out
class DeepSleep2(nn.Module):
    def __init__(self, num_filters=64, kernel_size=100, stride=4, num_residual_blocks=8, nb_classes=4):
        super(DeepSleep2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters,
                               kernel_size=kernel_size, stride=stride)

        self.maxpool = nn.MaxPool1d(2)
        self.resblocks = nn.Sequential(*[ResidualBlock2(num_filters, kernel_size, 1)
                                         for _ in range(num_residual_blocks)])
        self.maxpool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=num_filters, hidden_size=128, num_layers=3, 
                            batch_first=True, bidirectional=True)
        self.dense = nn.Linear(256, 256)

        self.batchnorm = nn.BatchNorm1d(256)
        self.classifier = nn.LazyLinear(nb_classes)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        print("After conv1:", out.shape)
        
        out = self.maxpool(out)
        print("After maxpool:", out.shape)
        
        out = self.resblocks(out)
        print("After resblocks:", out.shape)
        
        out = self.maxpool2(out)
        print("After maxpool2:", out.shape)
        
        res = self.adaptive_pool(out).squeeze(-1)  # Save maxpool output
        print("After adaptive_pool:", res.shape)
        
        out = out.transpose(1, 2)  # To match the LSTM's expected input
        out, _ = self.lstm(out)
        print("After LSTM:", out.shape)
        
        out = out[:, -1, :]  # Take the output from the final time step
        res = self.dense(res)  # adjust the residual to have the same number of features
        res = self.batchnorm(res)  # Apply batch normalization
        
        print("Before residual addition - out:", out.shape, ", res:", res.shape)
        
        out = out + res  # Add residual connection
        out = self.relu(out)
        out = self.classifier(out)
        return out 

# Create the model
#model = DeepSleep(num_filters=64, kernel_size=25, stride=1, num_residual_blocks=8)
#input_tensor = torch.randn(3, 1, 1920)
#output = model(input_tensor)
#print(output.shape) # This should be (3, 4)

#model = DeepSleep2(num_filters=64, kernel_size=100, stride=4, num_residual_blocks=8)
#input_tensor = torch.randn(3, 1, 7500)
#output = model(input_tensor)
#print(output.shape) # This should be (3, 4)

