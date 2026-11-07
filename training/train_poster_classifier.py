import argparse
from statistics import mean

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import torchvision.transforms.functional as Ft

from models.classification import CustomCNN

class ResizeAndPadTransform: #Pour le transform qd on load le dataset

    def __init__(self, target_size=224, padding="constant", fill=0):
        """
        Resize (grade le ratio) pour que la plus grande dimension de l'image fasse target_size
            et pad l'autre dimension pour obtenir une image carrée de target_size*target_size

        Parameters
        ----------
        target_size : int
            Taille cible demandée par le réseau utilisé.
        padding : str
            Type de padding à appliquer (constant, edge, reflect ou symmetric).
        fill : int
            Couleur du padding, 0 pour noir sur tous les channels.
        """
        self.target_size = target_size
        self.padding = padding
        self.fill = fill

    def __call__(self, img):
        """
        Parameters
        ----------
        img : PIL Image
            Image réelle du dataset.

        Returns
        -------
        Tensor
            Tenseur aux dimensions demandées par le réseau utilisé.
        """
        w, h = img.size
        max_dim = max(w, h)
        ratio = self.target_size / max_dim #pour garder le même ratio lors du resize
        new_w, new_h = int(w * ratio), int(h * ratio)
        img_resized = Ft.resize(img, (new_h, new_w)) #HEIGHT*WIDTH pour resize!

        p_left = (self.target_size - new_w) // 2   #pixels à rajouter à gauche
        p_right = self.target_size - new_w - p_left
        p_top = (self.target_size - new_h) // 2    #pixels à rajouter en haut
        p_bottom = self.target_size - new_h - p_top

        return Ft.pad(img_resized, (p_left, p_top, p_right, p_bottom), self.fill, self.padding)


def train(net, optimizer, loader, epochs=10, writer=None):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        running_loss = []
        t = tqdm(loader)
        for x, y in t:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'training loss: {mean(running_loss)}')
        writer.add_scalar('training loss', mean(running_loss), epoch)

def test(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x).argmax(1)
            test_corrects += y_hat.eq(y).sum().item()
            total += y.size(0)
    return test_corrects / total


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default = 'Poster_cls', help='experiment name')
    parser.add_argument('--batch_size', type=int, default = int(64), help='batch_size')
    parser.add_argument('--lr', type=float, default = float(1e-3), help='learning rate')
    parser.add_argument('--nb_epochs', type=int, default = int(10), help='number of epochs')
    parser.add_argument('--target_size', type=int, default = int(224), help='image size required by the model')
    parser.add_argument('--padding', type=str, default = "constant", help='type of padding (constant,edge,reflect,symmetric)')
    parser.add_argument('--fill', type=int, default = int(0), help='couleur du padding si "constant", 0=noir')
 
 
    args = parser.parse_args()
    exp_name = args.exp_name
    batch_size = args.batch_size
    nb_epochs = args.nb_epochs
    lr = args.lr
    target_size = args.target_size
    padding = args.padding
    fill = args.fill
 
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    # transforms
    transform = transforms.Compose([
        ResizeAndPadTransform(target_size=target_size, padding=padding, fill=fill),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalisation ImageNet
    ])
 
    # datasets
    dataset_path = '../content/sorted_movie_posters_paligema'   #! Faut se mettre d'accord sur un path précis et plus le changer
    full_dataset = torchvision.datasets.ImageFolder(root=dataset_path, transform=transform)

    # Séparation Train-Test (80% train, 20% test)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    # dataloaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
 
    net = CustomCNN().to(device)
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(f'runs/{exp_name}')
    
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
 
    train(net, optimizer, trainloader, nb_epochs, writer)
    test_acc = test(net, testloader)
 
    print(f'test accuracy: {test_acc}')
 
    # save your model
    os.makedirs('./weights', exist_ok=True)
 
    torch.save(net.state_dict(), f'./weights/{exp_name}.pth')
