from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms as T
import torchvision.transforms.functional as Ft
import torch.nn as nn
import torch.nn.functional as F

#! TODO :
#   - CustomCNN:
#       * Tester train loop
#       * Tester performancce avec différents paddings
#   - Transfer Learning efficientnet_b0:
#       * Créer une tête de classif à 10 classes
#       * Charger efficientnet_b0 avec les poids ImageNet
#       * Entraîner la tête avec efficientnet_b0 en backbone (frozen entièrement)
#       * Fine-tune la tête (avec un plus gros lr) et les dernières couches du backbone (petit lr)




class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 28 * 28, 512) # 224/2 /2 /2 = 28
        self.fc2 = nn.Linear(512, 10)
 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.max_pool2d(x, (2,2))
        x = F.relu(self.conv2(x))
        x = torch.max_pool2d(x, (2,2))
        x = F.relu(self.conv3(x))
        x = torch.max_pool2d(x, (2,2))
        x = torch.flatten(x, 1) #or x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x) #CrossEntropyLoss applique deja un softmax de base
        return x

    

