"""
Pretraining functions for classifier and GAN
"""
import os

import torch
import torch.nn as nn
import torch.optim as optim

from module.models.models import ConvNet
from module.training.training_step import classifier_train_step
from module.util.metrics import accuracy

def cls_pretraining(loader_train, loader_test, learning_rate, n_epochs, n_classes, device, results_path):
    """

    Args:
        loader_train (TYPE): 
        loader_test (TYPE): 
        learning_rate (TYPE): 
        n_epochs (TYPE): 
        n_classes (TYPE): 
        device (TYPE): 
        results_path (TYPE): 

    Returns:
        Trained classifier on loader
    """
    # Pretraining paths
    pretraining_path = ''.join([results_path, '/pretraining'])
    cls_pretrained_path = ''.join([pretraining_path, '/cls_pretrained.pth'])

    classifieur = ConvNet(classes=n_classes).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(classifieur.parameters(), lr=learning_rate)

    os.makedirs(pretraining_path, exist_ok=True)

    if os.path.isfile(cls_pretrained_path):
        classifieur.load_state_dict(torch.load(cls_pretrained_path))
        print('loaded existing model')

    else:
        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for batch_idx, (inputs, labels) in enumerate(loader_train):

                # data to gpu
                inputs, labels = inputs.to(device), labels.to(device)

                loss = classifier_train_step(classifieur, inputs, optimizer, criterion, labels)

                running_loss += loss

            print(f'Epoch: {epoch} || loss: {running_loss}')
            if (epoch + 1) % 10 == 0:
                print(f'Test accuracy: {100*accuracy(classifieur, loader_test):.2f}%')

    print('Finished Training')
    print('\n')
    print('Results:')
    print(f'Test accuracy: {100*accuracy(classifieur, loader_test):.2f}%')
    torch.save(classifieur.state_dict(), cls_pretrained_path)

    return classifieur
