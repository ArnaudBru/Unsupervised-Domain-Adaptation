import numpy as np

import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt

def data_sample(dataloader, sample_size=5):
  f, ax = plt.subplots(1,sample_size, figsize=(2*sample_size,4))

  for batch_idx, (data, target) in enumerate(dataloader):
      size = data.size()[-2:]
      for i, image in enumerate(data[:sample_size]):
        ax[i].imshow(-1*image.reshape(size).numpy(), cmap='Greys')
        ax[i].title.set_text(f'True Label: {target[i]}')
      break

def generated_sample(generator, n_classes, latent_dim):
  f, ax = plt.subplots(1,n_classes, figsize=(15,15))

  device = next(generator.parameters()).device
  generator.eval()

  z = Variable(torch.randn(n_classes, latent_dim)).to(device)
  gen_labels = Variable(torch.LongTensor(np.arange(n_classes))).to(device)

  gen_imgs = generator(z, gen_labels).view(-1,1,img_size, img_size)
  size = gen_imgs.size()[-2:]

  for i, image in enumerate(gen_imgs):
    image = image.cpu().detach()
    ax[i].imshow(-1*image.numpy().reshape(size), cmap='Greys')
    ax[i].set_title(gen_labels[i].item())
  plt.show()

