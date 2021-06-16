"""
Pretraining functions for classifier and GAN
"""
import os

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from torchvision.utils import save_image

from training.training_step import classifier_train_step, generator_train_step, discriminator_train_step
from util.metrics import accuracy


def cls_pretraining(classifier, loader_train, loader_test, learning_rate, n_epochs, results_path):
    """

    Args:
        classifier (TYPE):
        loader_train (TYPE):
        loader_test (TYPE):
        learning_rate (TYPE):
        n_epochs (TYPE):
        results_path (TYPE):

    Returns:
        Trained classifier on loader
    """

    # Pretraining paths
    cls_pretrained_path = ''.join([results_path, '/cls_pretrained.pth'])

    device = next(classifier.parameters()).device

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    os.makedirs(results_path, exist_ok=True)

    if os.path.isfile(cls_pretrained_path):
        classifier.load_state_dict(torch.load(cls_pretrained_path))
        print('loaded existing model')

    else:
        print('Starting Training classifier')
        for epoch in range(n_epochs):  # loop over the dataset multiple times
            running_loss = 0.0

            for inputs, labels in loader_train:
                inputs, labels = inputs.to(device), labels.to(device)
                loss = classifier_train_step(classifier, inputs, optimizer, criterion, labels)
                running_loss += loss

            print(f'Epoch: {epoch} || loss: {running_loss}')
            if (epoch + 1) % 10 == 0:
                print(f'Test accuracy: {100*accuracy(classifier, loader_test):.2f}%')
        print('Finished Training classifier')
        print('\n')

    print('Results:')
    print(f'Test accuracy: {100*accuracy(classifier, loader_test):.2f}%')
    torch.save(classifier.state_dict(), cls_pretrained_path)

    return classifier


def gan_pretraining(generator, discriminator, classifier, loader_train,
                    lr_g, lr_d, latent_dim, n_classes, n_epochs,
                    img_size, results_path):
    """
    Args:
        generator (TYPE)
        discriminator (TYPE)
        classifier (TYPE)
        loader_train (TYPE)
        lr_g (TYPE)
        lr_d (TYPE)
        latent_dim (TYPE)
        n_classes (TYPE)
        n_epochs (TYPE)
        img_size (TYPE)
        results_path (TYPE)

    Returns:
        Pretrained generator and discriminator
    """
    img_pretraining_path = ''.join([results_path, '/images'])
    models_pretraining_path = ''.join([results_path, '/gan_models'])

    g_pretrained = ''.join([results_path, '/generator_pretrained.pth'])
    d_pretrained = ''.join([results_path, '/discriminator_pretrained.pth'])

    device = next(classifier.parameters()).device

    criterion_gan = nn.BCELoss().to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g)

    os.makedirs(img_pretraining_path, exist_ok=True)
    os.makedirs(models_pretraining_path, exist_ok=True)

    loaded_gen = False
    loaded_dis = False

    if os.path.isfile(g_pretrained):
        generator.load_state_dict(torch.load(g_pretrained))
        print('loaded existing generator')
        loaded_gen = True

    if os.path.isfile(d_pretrained):
        discriminator.load_state_dict(torch.load(d_pretrained))
        print('loaded existing discriminator')
        loaded_dis = True

    if not(loaded_gen and loaded_dis):
        print('Starting Training GAN')
        for epoch in range(n_epochs):
            print(f'Starting epoch {epoch}/{n_epochs}...', end=' ')
            g_loss_list = []
            d_loss_list = []
            for i, (images, _) in enumerate(loader_train):

                real_images = Variable(images).to(device)
                _, labels = torch.max(classifier(real_images), dim=1)

                generator.train()

                d_loss = discriminator_train_step(discriminator, generator, d_optimizer, criterion_gan,
                                                  real_images, labels, latent_dim, n_classes)
                d_loss_list.append(d_loss)

                g_loss = generator_train_step(discriminator, generator, g_optimizer, criterion_gan,
                                              loader_train.batch_size, latent_dim, n_classes=n_classes)
                g_loss_list.append(g_loss)

            generator.eval()

            latent_space = Variable(torch.randn(n_classes, latent_dim)).to(device)
            gen_labels = Variable(torch.LongTensor(np.arange(n_classes))).to(device)

            gen_imgs = generator(latent_space, gen_labels).view(-1, 1, img_size, img_size)
            save_image(gen_imgs.data, img_pretraining_path + f'/epoch_{epoch:02d}.png', nrow=n_classes, normalize=True)
            torch.save(generator.state_dict(), models_pretraining_path + f'/{epoch:02d}_gen.pth')
            torch.save(discriminator.state_dict(), models_pretraining_path + f'/{epoch:02d}_dis.pth')

            print(f"[D loss: {np.mean(d_loss_list)}] [G loss: {np.mean(g_loss_list)}]")
        print('Finished Training GAN')
        print('\n')


    return generator, discriminator
