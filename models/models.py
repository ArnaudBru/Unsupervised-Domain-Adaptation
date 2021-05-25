import torch.nn as nn
import torch.nn.functional as F

class convNet(nn.Module):
    def __init__(self, channels=1, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size= 5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size= 5)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x



class Generator(nn.Module):
    def __init__(self, classes=10, latent_dim=100, size=28):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = size
        
        self.label_emb = nn.Embedding(classes, classes)
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim + classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.img_size**2),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), self.latent_dim)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), self.img_size, self.img_size)



class Discriminator(nn.Module):
    def __init__(self, classes=10, size=28):
        super().__init__()
        
        self.img_size = size

        self.label_emb = nn.Embedding(classes, classes)
        
        self.model = nn.Sequential(
            nn.Linear(self.img_size**2 + classes, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), self.img_size**2)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()