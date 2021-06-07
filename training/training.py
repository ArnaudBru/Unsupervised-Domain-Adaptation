import numpy as np

import torch
from torch.autograd import Variable


def classifier_train_step(classifieur, inputs, optimizer, criterion, images, labels):
    optimizer.zero_grad()
    outputs = classifieur(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, latent_dim, labels=None, n_classes=None):
    g_optimizer.zero_grad()

    device = next(generator.parameters()).device

    z = Variable(torch.randn(batch_size, latent_dim)).to(device)
    # If no labels are given we generate random labels
    if labels is None:
        assert isinstance(n_classes, int), 'n_classes must be of type int when labels are not given'
        labels = Variable(torch.LongTensor(np.random.randint(0, n_classes, batch_size))).to(device)
    fake_images = generator(z, labels)

    validity = discriminator(fake_images, labels)

    g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()

def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels, latent_dim, n_classes):
    d_optimizer.zero_grad()

    device = next(generator.parameters()).device

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))
    
    # train with fake images
    z = Variable(torch.randn(batch_size, latent_dim)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, n_classes, batch_size))).to(device)
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()
