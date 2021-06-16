"""
Implementation of Generative Pseudo-label Refinement for Unsupervised Domain Adaptation
Paper: Pietro Morerio, Riccardo Volpi, Ruggero Ragonesi, Vittorio Murino
https://arxiv.org/pdf/2001.02950.pdf

Implementation by: Arnaud Brugiere
https://github.com/ArnaudBru

Attributes:
    BATCH_SIZE (int):
    classifieur (TYPE):
    cls_criterion (TYPE):
    cls_optimizer (TYPE):
    d_optimizer (TYPE):
    DATA_PATH (str):
    DEVICE (TYPE):
    discriminator (TYPE):
    g_optimizer (TYPE):
    gan_criterion (TYPE):
    generator (TYPE):
    IMG_SIZE (int):
    IMG_TRAINING_PATH (TYPE):
    LATENT_DIM (int):
    LR_CLS_PRETRAINING (float):
    LR_CLS_TRAINING (float):
    LR_D_PRETRAINING (float):
    LR_D_TRAINING (float):
    LR_G_PRETRAINING (float):
    LR_G_TRAINING (float):
    MODELS_TRAINING_PATH (TYPE):
    N_CLASSES (int):
    N_EPOCHS_CLS_PRETRAINING (int):
    N_EPOCHS_GAN_PRETRAINING (int):
    N_EPOCHS_TRAINING (int):
    PRETRAINING_PATH (TYPE):
    RESULTS_PATH (str):
    TRAINING_PATH (TYPE):
    VERBOSE (bool):
"""
import os

import numpy as np

from torchvision.utils import save_image

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from util.metrics import accuracy, class_accuracy
from util.util import generated_sample
from models.models import ConvNet, Generator, Discriminator
from data.datasets import load_mnist, load_svhn
from training.training_step import classifier_train_step, generator_train_step, discriminator_train_step
from training.pretraining import cls_pretraining, gan_pretraining

# --------------------
#     Parameters
# --------------------

# Data Loading parameters
DATA_PATH = './datasets'
IMG_SIZE = 28
BATCH_SIZE = 32

VERBOSE = False

# GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes
N_CLASSES = 10

# Parameters classifier pretraining
N_EPOCHS_CLS_PRETRAINING = 25
# Learning rate classifier pretraining
LR_CLS_PRETRAINING = 1e-4

# Latent space dimension
LATENT_DIM = 100
# Learning rate discriminator pretraining
LR_D_PRETRAINING = 1e-4
# Learning rate generator pretraining
LR_G_PRETRAINING = 1e-4
# Parameters classifier pretraining
N_EPOCHS_GAN_PRETRAINING = 30

# Learning rate classifier pretraining
LR_CLS_TRAINING = 1e-4
# Learning rate discriminator training
LR_D_TRAINING = 1e-4
# Learning rate generator training
LR_G_TRAINING = 1e-4
# Parameters classifier training
N_EPOCHS_TRAINING = 30

# Results
RESULTS_PATH = './results'

# Pretraining paths
PRETRAINING_PATH = ''.join([RESULTS_PATH, '/pretraining'])

# Training paths
TRAINING_PATH = ''.join([RESULTS_PATH, '/training'])
IMG_TRAINING_PATH = ''.join([TRAINING_PATH, '/images'])
MODELS_TRAINING_PATH = ''.join([TRAINING_PATH, '/models'])

# --------------------
#   Data loading
# --------------------


mnist_loader, mnist_loader_test = load_mnist(DATA_PATH, IMG_SIZE, BATCH_SIZE)
svhn_loader_train, svhn_loader_test = load_svhn(DATA_PATH, IMG_SIZE, BATCH_SIZE)


# --------------------
#   Pretraining
# --------------------

# Classifier pretraining on source data
classifieur = ConvNet(classes=N_CLASSES).to(DEVICE)
classifieur = cls_pretraining(classifieur, svhn_loader_train, svhn_loader_test,
                              LR_CLS_PRETRAINING, N_EPOCHS_CLS_PRETRAINING, PRETRAINING_PATH)

if VERBOSE:
    class_accuracy(classifieur, svhn_loader_test, list(range(N_CLASSES)))

# GAN pretraining on target data annotated by classifier
generator = Generator(N_CLASSES, LATENT_DIM, IMG_SIZE).to(DEVICE)
discriminator = Discriminator(N_CLASSES, IMG_SIZE).to(DEVICE)

generator, discriminator = gan_pretraining(generator, discriminator,
                                           classifieur, mnist_loader,
                                           LR_G_PRETRAINING, LR_D_PRETRAINING,
                                           LATENT_DIM, N_CLASSES,
                                           N_EPOCHS_GAN_PRETRAINING,
                                           IMG_SIZE, PRETRAINING_PATH)

if VERBOSE:
    generated_sample(generator, N_CLASSES, LATENT_DIM, IMG_SIZE)

    print('Baseline Algorithm')
    print(f'Test accuracy: {100*accuracy(classifieur, mnist_loader_test):.2f}%')
    class_accuracy(classifieur, mnist_loader_test, list(range(N_CLASSES)))


# --------------------
#     Training
# --------------------


# Classifier loss and optimizer
cls_criterion = nn.CrossEntropyLoss().to(DEVICE)
cls_optimizer = optim.Adam(classifieur.parameters(), lr=LR_CLS_TRAINING)

# GAN loss and optimizers
gan_criterion = nn.BCELoss().to(DEVICE)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR_D_TRAINING)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR_G_TRAINING)

# Create folder for models and generated images
os.makedirs(IMG_TRAINING_PATH, exist_ok=True)
os.makedirs(MODELS_TRAINING_PATH, exist_ok=True)

# Training
print('Starting Training GAN')
for epoch in range(N_EPOCHS_TRAINING):
    print(f'Starting epoch {epoch}/{N_EPOCHS_TRAINING}...', end=' ')
    g_loss_list = []
    d_loss_list = []
    c_loss_list = []
    for i, (images, _) in enumerate(mnist_loader):

        generator.train()

        # Step 3
        # Sample latent space and random labels
        z = Variable(torch.randn(BATCH_SIZE, LATENT_DIM)).to(DEVICE)
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, N_CLASSES, BATCH_SIZE))).to(DEVICE)

        # Step 4
        # Generate fake images
        fake_images = generator(z, fake_labels).view(BATCH_SIZE, 1, IMG_SIZE, IMG_SIZE)

        # Step 5
        # Update classifier
        c_loss = classifier_train_step(classifieur, fake_images, cls_optimizer, cls_criterion, fake_labels)
        c_loss_list.append(c_loss)

        # Step 6
        # Sample real images from target data
        real_images = Variable(images).to(DEVICE)

        # Step 7
        # Infer labels using classifier
        _, labels_classifier = torch.max(classifieur(real_images), dim=1)

        # Step 8
        # Update discriminator
        d_loss = discriminator_train_step(discriminator, generator, d_optimizer, gan_criterion,
                                          real_images, labels_classifier, LATENT_DIM, N_CLASSES)
        d_loss_list.append(d_loss)

        # Step 9
        # Update Generator
        g_loss = generator_train_step(discriminator, generator, g_optimizer, gan_criterion,
                                      BATCH_SIZE, LATENT_DIM, labels=labels_classifier)
        g_loss_list.append(g_loss)

    generator.eval()

    z = Variable(torch.randn(N_CLASSES, LATENT_DIM)).to(DEVICE)
    gen_labels = Variable(torch.LongTensor(np.arange(N_CLASSES))).to(DEVICE)

    gen_imgs = generator(z, gen_labels).view(-1, 1, IMG_SIZE, IMG_SIZE)
    save_image(gen_imgs.data, IMG_TRAINING_PATH + f'/epoch_{epoch:02d}.png', nrow=N_CLASSES, normalize=True)
    torch.save(classifieur.state_dict(), MODELS_TRAINING_PATH + f'/{epoch:02d}_cls.pth')
    torch.save(generator.state_dict(), MODELS_TRAINING_PATH + f'/{epoch:02d}_gen.pth')
    torch.save(discriminator.state_dict(), MODELS_TRAINING_PATH + f'/{epoch:02d}_dis.pth')

    print(f"[D loss: {np.mean(d_loss_list)}] [G loss: {np.mean(g_loss_list)}] [C loss: {np.mean(c_loss_list)}]")
print('Finished Training GAN')
print('\n')


# --------------------
#     Final Model
# --------------------


print(f'MNIST accuracy: {100*accuracy(classifieur, mnist_loader_test):.2f}%')
class_accuracy(classifieur, mnist_loader_test, list(range(N_CLASSES)))
if VERBOSE:
    generated_sample(generator, N_CLASSES, LATENT_DIM, IMG_SIZE)
