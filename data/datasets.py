from torchvision.datasets import MNIST, SVHN
from torchvision import transforms
from torch.utils.data import DataLoader

def load_mnist(data_path, img_size=28, batch_size=32):

    mnist_dataset_train = MNIST(root=data_path, train=True,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((img_size, img_size)),
                                                        transforms.Normalize(mean=[0.5], std=[0.5])
                                                        ]),
                          download=True)

    mnist_dataset_test = MNIST(root=data_path, train=False,
                          transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((img_size, img_size)),
                                                        transforms.Normalize(mean=[0.5], std=[0.5])
                                                        ]),
                          download=True)

    mnist_loader_train = DataLoader(dataset=mnist_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    mnist_loader_test = DataLoader(dataset=mnist_dataset_test, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)

    return mnist_loader_train, mnist_loader_test