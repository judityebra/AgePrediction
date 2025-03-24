import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.utils import return_paths, task_importance_weights, df_loader, cost_fn, compute_mae_and_mse, plot_losses
from models.models import resnet34
from torch.utils.data.dataloader import default_collate
import argparse


def predict_age(image_path, model_path, device, loss, num_classes, add_class, grayscale = False):
    """
    Realitza una predicció de l'edat a partir d'una imatge i un model preentrenat.

    Parameters:
    - image_path (str): Path d'on es troba la imatge.
    - model_path (str): Path d'on es troba el model entrenat.
    - device (torch.device): Dispositiu on es carregarà el model (CPU o GPU).
    - loss (str): Tipus de pèrdua utilitzada.
    - num_classes (int): Nombre de classes per a la classificació.
    - add_class (int): Valor a afegir a la predicció per ajustar-la a l'edat mínima.
    - grayscale (bool, opcional): Indica si l'imatge d'entrada és en escala de grisos.
    """

    # Transformacions a la imatge
    img = Image.open(image_path).convert('RGB')
    img = custom_transform(img)
    img = img.unsqueeze(0)  

    # Carregar model entrenat
    model = resnet34(num_classes, grayscale, loss)  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.set_grad_enabled(False):
        img = img.to(device)
        logits, probas = model(img)
        predict_levels = probas > 0.5
        if loss.lower() == "ordinal" or "coral":
            predicted_label = torch.sum(predict_levels, dim=1)
            print('Predicted age in years:', predicted_label.item() + add_class)
        elif loss.lower() == "ce":
            print('Predicted age in years:', torch.argmax(probas, 1).item() + add_class)
        else:
            raise ValueError("Loss mal introduïda")

        


if __name__ == "__main__":
    ########## ARGUMENTS #############
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',
                        type=int,
                        default=-1)

    parser.add_argument('--numworkers',
                        type=int,
                        default=6)

    parser.add_argument('--loss',
                        type=str,
                        default='coral')

    parser.add_argument('--state_dict_path',
                        type=str)
    
    parser.add_argument('--image_path',
                        type=str)
    
    parser.add_argument('--dataset',
                        type=str,
                        default='CACD')

    args = parser.parse_args()

    NUM_WORKERS = args.numworkers
    LOSS = args.loss

    if args.cuda >= 0:
        DEVICE = torch.device("cuda:%d" % args.cuda)
    else:
        DEVICE = torch.device("cpu")

    IMAGE_PATH = args.image_path
    STATE_DICT_PATH = args.state_dict_path

    DATASET = args.dataset
    if DATASET.upper() == 'CACD':
        NUM_CLASSES = 49
        ADD_CLASS = 14
    elif DATASET.upper() == 'AFAD':
        NUM_CLASSES = 26
        ADD_CLASS = 15
    else:
        raise ValueError("Incorrect dataset introduced.")

    # Transformació de la imatge
    custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                            transforms.RandomCrop((120, 120)),
                                            transforms.ToTensor()])

    # Predir l'edat 
    predicted_age = predict_age(IMAGE_PATH, STATE_DICT_PATH, DEVICE, LOSS, NUM_CLASSES, ADD_CLASS)


