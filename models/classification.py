from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

###? IDEE : Fine-tuner le modele efficientnet_b0 (petit, rapide et avec de bons résultats en classification d'après mes recherches) sur nos affiches de films
##? ===> ATTENTION : il s'attend à des images carrés (224x224)!

#! TODO :
#   - Traiter les affiches:
#       * conserver le ratio
#       * resize pour que la plus grande dimension fasse 224
#       * ajout de padding sur la plus petite dimension pour obtenir un carré 224x224 comme attendu (tester Padding noir et reflect padding ?)
#       * ATTENTION : avoir une normalisation ImageNet aussi!!
#   - Créer une tête de classif à 10 classes
#   - Charger efficientnet_b0 avec les poids ImageNet
#   - Entraîner la tête avec efficientnet_b0 en backbone (frozen entièrement)
#   - Fine-tune la tête (avec un plus gros lr) et les dernières couches du backbone (petit lr)

