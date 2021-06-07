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

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion, latent_dim, device, labels=None, n_classes=None):
    g_optimizer.zero_grad()

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
    