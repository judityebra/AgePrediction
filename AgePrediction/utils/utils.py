# Definim les llibreries que utilitzarem
import os
import torch
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Definim la classe DatasetAge que hereta de Dataset
class DatasetAge(Dataset):
    """
    Dataset personalitzat per carregar imatges de cares amb edats

    Parameters:
    - csv_path (str): Ruta al fitxer CSV amb les dades.
    - img_dir (str): Directori de les imatges.
    - loss (str): Tipus de pèrdua ('ce', 'coral', 'ordinal').
    - num_classes (int): Nombre de classes.
    - dataset (str): Nom del dataset ('AFAD' o 'CACD').
    - transform (callable, opcional): Transformacions a aplicar a les imatges.

    """
    def __init__(self, csv_path, img_dir, loss, num_classes, dataset, transform=None):
        # Llegim el CSV amb les dades
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.dataset = dataset
        self.csv_path = csv_path
        # Segons el dataset, assignem el camí de les imatges o fem que salti un error
        if self.dataset == 'AFAD':
            self.img_paths = df['path']
        elif self.dataset == 'CACD':
            self.img_paths = df['file'].values
        else:
            raise ValueError("Incorrect model name introduced.")
        self.y = df['age'].values
        self.transform = transform
        self.loss = loss
        self.NUM_CLASSES = num_classes

    def __getitem__(self, index):
        # Obtenim el camí de la imatge i la carreguem
        img_path = os.path.join(self.img_dir, self.img_paths[index])
        try:
            img = Image.open(img_path)
        except FileNotFoundError:
            # print(f"File not found: {img_path}") #print que es va fer de prova per mirar quines imatges no es carregaven
            return None, None, None

        # Apliquem la transformació si està definida
        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        if self.loss != 'ce':
            # Si la pèrdua (loss) no és entropia creuada (ce), generem els nivells de sortida
            levels = [1] * label + [0] * (self.NUM_CLASSES - 1 - label)
            levels = torch.tensor(levels, dtype=torch.float32)
            return img, label, levels

        return img, label

    def __len__(self):
        return self.y.shape[0]

# Funció per retornar les rutes dels fitxers segons el dataset
def return_paths(df):
    """
    Retorna les rutes dels fitxers segons el dataset.

    Parameters:
    - df (str): Nom del dataset ('AFAD' o 'CACD').

    Returns:
    - list: Llista amb les rutes dels fitxers.
    """
    ll_df = []
    if df.lower() == 'cacd':
        ll_df.append('./../coral-cnn-master/datasets/cacd_train.csv')  # ruta entrenament
        ll_df.append('./../coral-cnn-master/datasets/cacd_valid.csv')  # ruta validació
        ll_df.append('./../coral-cnn-master/datasets/cacd_test.csv')   # ruta test
        ll_df.append('./../coral-cnn-master/datasets/CACD2000-centered')  # ruta imatges
        return ll_df
    elif df.lower() == 'afad':
        ll_df.append('./coral-cnn-master/datasets/afad_train.csv')  # ruta entrenament
        ll_df.append('./coral-cnn-master/datasets/afad_valid.csv')  # ruta validació
        ll_df.append('./coral-cnn-master/datasets/afad_test.csv')   # ruta test
        ll_df.append('./coral-cnn-master/datasets/tarball-master/AFAD-Full')  # ruta imatges
        return ll_df
    else:
        raise ValueError("Incorrect dataset introduced.")

# Funció per calcular els pesos d'importància per la pèrdua ordinal
def task_importance_weights(label_array, imp_weight, num_classes):
    """
    Calcula els pesos d'importància per a la pèrdua ordinal.

    Parameters:
    - label_array (tensor): Tensor amb les edats.
    - imp_weight (int): Pes d'importància (0 o 1).
    - num_classes (int): Nombre de classes.

    Returns:
    - tensor: Pesos d'importància per a la pèrdua ordinal.
    """
    if not imp_weight:
        imp = torch.ones(num_classes - 1, dtype=torch.float)
        return imp
    elif imp_weight == 1:
        uniq = torch.unique(label_array)
        num_examples = label_array.size(0)
        m = torch.zeros(uniq.shape[0])
        for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
            m_k = torch.max(torch.tensor([label_array[label_array > t].size(0),
                                          num_examples - label_array[label_array > t].size(0)]))
            m[i] = torch.sqrt(m_k.float())
        imp = m / torch.max(m)
        imp = imp[0:num_classes - 1]
        return imp
    else:
        raise ValueError('Incorrect importance weight parameter.')

# Funció per carregar els datasets de tren, validació i test
def df_loader(train_p, valid_p, test_p, image_p, batch_size, n_workers, loss_dataset, num_classes, dataset, collate_fn):
    """
    Carrega els datasets de tren, validació i test.

    Parameters:
    - train_p (str): Ruta del fitxer CSV de tren.
    - valid_p (str): Ruta del fitxer CSV de validació.
    - test_p (str): Ruta del fitxer CSV de test.
    - image_p (str): Directori de les imatges.
    - batch_size (int): Mida del batch.
    - n_workers (int): Nombre de treballadors per a la càrrega de dades.
    - loss_dataset (str): Tipus de pèrdua.
    - num_classes (int): Nombre de classes.
    - dataset (str): Nom del dataset.
    - collate_fn (callable): Funció de col·lecció personalitzada.

    Returns:
    - tuple: Conté els carregadors de tren, validació i test, i la longitud del dataset de tren.
    """
    custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                           transforms.RandomCrop((120, 120)),
                                           transforms.ToTensor()])

    train_dataset = DatasetAge(csv_path=train_p,
                               img_dir=image_p,
                               loss=loss_dataset, num_classes=num_classes, dataset=dataset,
                               transform=custom_transform)

    custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                            transforms.CenterCrop((120, 120)),
                                            transforms.ToTensor()])

    test_dataset = DatasetAge(csv_path=test_p,
                              img_dir=image_p,
                              loss=loss_dataset, num_classes=num_classes, dataset=dataset,
                              transform=custom_transform2)

    valid_dataset = DatasetAge(csv_path=valid_p,
                               img_dir=image_p,
                               loss=loss_dataset, num_classes=num_classes, dataset=dataset,
                               transform=custom_transform2)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=n_workers,
                              collate_fn=collate_fn)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=n_workers,
                              collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=n_workers,
                             collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader, len(train_dataset)

# Funció per calcular la funció de pèrdua (loss function) segons el model
def cost_fn(nom_model, logits=None, levels=None, imp=None, targets=None):
    """
    Calcula la funció de pèrdua segons el model.

    Parameters:
    - nom_model (str): Nom del model ('ce', 'coral', 'ordinal').
    - logits (tensor, opcional): Sortida del model.
    - levels (tensor, opcional): Nivells per a la pèrdua ordinal.
    - imp (tensor, opcional): Pesos d'importància.
    - targets (tensor, opcional): Etiquetes reals.

    Returns:
    - tensor: Valor de la pèrdua.
    """
    if nom_model == 'ce':
        return F.cross_entropy(logits, targets)
    if nom_model == 'coral':
        val = (-torch.sum((F.logsigmoid(logits) * levels
                           + (F.logsigmoid(logits) - logits) * (1 - levels)) * imp,
                          dim=1))
        return torch.mean(val)
    if nom_model == 'ordinal':
        val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1] * levels
                           + F.log_softmax(logits, dim=2)[:, :, 0] * (1 - levels)) * imp, dim=1))
        return torch.mean(val)
    else:
        raise ValueError('ERROR EN LA TRIA DE MODEL (cost_fn)')

# Funció per calcular MAE i MSE
def compute_mae_and_mse(model, data_loader, device, nom_model):
    """
    Calcula el Mean Absolute Error (MAE) i el Mean Squared Error (MSE) del model.
    El MAE (Mean Absolute Error) i l'MSE (Mean Squared Error) són dues mètriques comunes utilitzades 
    per avaluar el rendiment dels models de regressió. 
    El MAE és la mitjana de les diferències absolutes entre els valors predits pel model i els valors reals. 
    L'MSE és la mitjana de les diferències quadrades entre els valors predits pel model i els valors reals.

    Parameters:
    - model (nn.Module): El model entrenat.
    - data_loader (DataLoader): Carregador de dades.
    - device (torch.device): Dispositiu d'execució (CPU o CUDA).
    - nom_model (str): Nom del model ('ce', 'coral', 'ordinal').

    Returns:
    - tuple: MAE i MSE.
    """
    mae, mse, num_examples = 0, 0, 0
    for i, tupla in enumerate(data_loader):
        if tupla is None or tupla[0] is None:
            continue
        if nom_model == 'ce':
            features = tupla[0]
            targets = tupla[1]
            levels = None
        else:
            features = tupla[0]
            targets = tupla[1]
            levels = tupla[2]
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        if nom_model != 'ce':
            predict_levels = probas > 0.5
            predicted_labels = torch.sum(predict_levels, dim=1)
        else:
            _, predicted_labels = torch.max(probas, 1)

        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets) ** 2)

    if num_examples == 0:
        return float('nan'), float('nan')  # o retornar 0.0, 0.0 o qualsevol altre valor apropiat

    mae = float(mae) / num_examples
    mse = float(mse) / num_examples
    return mae, mse

# Funció per dibuixar les pèrdues d'entrenament
def plot_losses(l, model_name=""):
    """
    Dibuixa les pèrdues d'entrenament al llarg de les èpoques.

    Parameters:
    - l (list): Llista de pèrdues.
    - model_name (str, opcional): Nom del model.

    """
    plt.plot(l, label="training loss")
    plt.legend()
    loss_title = model_name
    plt.title(loss_title)

    plt.savefig('ce_loss.png')
