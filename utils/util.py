import matplotlib.pyplot as plt

def show_sample(dataloader, sample_size=5):
  f, ax = plt.subplots(1,sample_size, figsize=(3*sample_size,3*sample_size))

  for batch_idx, (data, target) in enumerate(mnist_loader):
      size = data.size()[-2:]
      for i, image in enumerate(data[:sample_size]):
        ax[i].imshow(-1*image.reshape(size).numpy(), cmap='Greys')
        ax[i].title.set_text(f'True Label: {target[i]}')
      break
