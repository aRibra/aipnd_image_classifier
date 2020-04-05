
# mnist multiplercepton

def sigmoid(x):
    return 1 / (1+np.exp(-x))

batch_size = 64

inputs_ = inputs_.view(batch_size, -1)

input_layer = torch.randn((int(28*28), 256))
bias_1 = torch.randn(256)

hidden = torch.randn((256, 10))
bias_2 = torch.randn(10)

out_hidden = sigmoid(torch.mm(inputs_, input_layer) + bias_1)

out = torch.mm(out_hidden, hidden) + bias_2

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x))

# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)


class MnistNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.hidden = nn.Linear(784, 256)
        self.output = nn.Linear(256, 10)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
        def forward(self, x):
            x = self.hidden(x)
            x = self.sigmoid(x)
            x = self.output(x)
            x = self.softmax(x)
            return x


class CustomNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_1 = nn.Linear(784, 128)
        self.hidden_2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.hidden_1(x)
        x = F.sigmoid(x)
        x = self.hidden_2(x)
        x = F.sigmoid(x)        
        x = self.output(x)
        x = F.softmax(x,  dim=1)
        
        return x
