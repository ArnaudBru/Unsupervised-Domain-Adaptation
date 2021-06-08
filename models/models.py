"""
Models Architectures
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    """Classifier CNN

    Attributes:
        classifier (TYPE): Classification using FC
        features (TYPE): Features extraction using Convolutions
    """

    def __init__(self, channels=1, classes=10):
        """
        Args:
            channels (int, optional): number of channels of the input image
            classes (int, optional): number of classes
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, classes),
        )

    def forward(self, inputs):
        """
        Args:
            inputs (TYPE): Input image

        Returns:
            TYPE: Softmax of logits
        """
        inputs = self.features(inputs)
        inputs = inputs.view(-1, 128 * 4 * 4)
        inputs = F.softmax(self.classifier(inputs), dim=1)
        return inputs


class Generator(nn.Module):

    """
    Attributes:
        img_size (TYPE): Input image size
        label_emb (TYPE): label embedding
        latent_dim (TYPE): dimension of the latent space of the generator
        model (TYPE): generator
    """

    def __init__(self, classes=10, latent_dim=100, size=28):
        """
        Args:
            classes (int, optional): number of classes
            latent_dim (int, optional): dimension of the latent space of the generator
            size (int, optional): Input image size
        """
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

    def forward(self, latent_space, labels):
        """
        Args:
            latent_space (TYPE): latent space input
            labels (TYPE): labels input

        Returns:
            TYPE: generated image
        """
        latent_space = latent_space.view(latent_space.size(0), self.latent_dim)
        labels = self.label_emb(labels)
        inputs = torch.cat([latent_space, labels], 1)
        out = self.model(inputs)
        return out.view(inputs.size(0), self.img_size, self.img_size)


class Discriminator(nn.Module):

    """
    Attributes:
        img_size (TYPE): image size
        label_emb (TYPE): label embedding
        model (TYPE):
    """

    def __init__(self, classes=10, size=28):
        """
        Args:
            classes (int, optional): number of classes
            size (int, optional): image size
        """
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

    def forward(self, images, labels):
        """
        Args:
            images (TYPE): images input
            labels (TYPE): labels input

        Returns:
            TYPE: Classification label
        """
        images = images.view(images.size(0), self.img_size**2)
        labels = self.label_emb(labels)
        inputs = torch.cat([images, labels], 1)
        out = self.model(inputs)
        return out.squeeze()
