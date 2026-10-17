from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import torchvision.transforms as T
import torchvision.transforms.functional as F

#! TODO :
#   - Traiter les affiches:
#       * conserver le ratio
#       * resize pour que la plus grande dimension fasse 224
#       * ajout de padding sur la plus petite dimension pour obtenir un carré 224x224 comme attendu (tester Padding noir et reflect padding ?)
#       * ATTENTION : avoir une normalisation ImageNet aussi!! -> mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#   - Créer une tête de classif à 10 classes
#   - Charger efficientnet_b0 avec les poids ImageNet
#   - Entraîner la tête avec efficientnet_b0 en backbone (frozen entièrement)
#   - Fine-tune la tête (avec un plus gros lr) et les dernières couches du backbone (petit lr)


def ResizeAndPad(img, target_size=224, padding="constant", fill=0):
    """
    Resize (grade le ratio) pour que la plus grande dimension de l'image fasse target_size
            et pad l'autre dimension pour obtenir une image carrée de target_size*target_size

    Parameters
    ----------
    img : PIL Image
        Image réelle du dataset.
    target_size : int
        Taille cible demandée par le réseau utilisé.
    padding : str
        Type de padding à appliquer (constant, edge, reflect ou symmetric).
    fill : int
        Couleur du padding, 0 pour noir sur tous les channels.

    Returns
    -------
    Tensor
        Tensor aux dimensions demandées par le réseau utilisé.
    """

    w,h = img.size

    max_dim = max(w,h)
    ratio = target_size/max_dim #pour garder le même ratio lors du resize
    new_w, new_h = int(w*ratio),int(h*ratio)
    img_resized = F.resize(img, (new_h,new_w)) #HEIGHT*WIDTH pour resize!

    p_left = (target_size - new_w) // 2 #pixels à rajouter à gauche
    p_right = target_size - new_w - p_left
    p_top = (target_size - new_h) // 2 #pixels à rajouter en haut
    p_bottom = target_size - new_h - p_top

    return F.pad(img_resized, (p_left, p_top, p_right, p_bottom), fill, padding)

from PIL import Image

img = Image.open("../../content/sorted_movie_posters_paligema/romance/308.jpg")
img.show()

pad_img = ResizeAndPad(img, target_size=224, padding="edge", fill=0)

pad_img.show()

    

