import random
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from tqdm import tqdm
import os

'''
MODIFIED TRAINING CODE
One new change:
1. split train into train + validation
2. tune threshold on validation for best F1-macro
'''

# --------------------------------------------------
# Paths
# --------------------------------------------------
OR_PATH = os.getcwd()
os.chdir("..")
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep
os.chdir(OR_PATH)

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------
n_epoch = 10
BATCH_SIZE = 30
LR = 0.0001
IMAGE_SIZE = 224
CHANNELS = 3

NICKNAME = "Callisto"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True

# this is only used as an initial value
THRESHOLD = 0.5

# validation split
VAL_SIZE = 0.2
RANDOM_STATE = 42

# thresholds to search
THRESHOLD_LIST = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]


# --------------------------------------------------
# Optional old CNN
# --------------------------------------------------
class CNN(nn.Module):
    def __init__(self, outputs_a):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.pad1 = nn.ZeroPad2d(2)

        self.conv2 = nn.Conv2d(16, 128, (3, 3))
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.linear = nn.Linear(128, outputs_a)
        self.act = torch.relu

    def forward(self, x):
        x = self.pad1(self.convnorm1(self.act(self.conv1(x))))
        x = self.act(self.conv2(self.act(x)))
        return self.linear(self.global_avg_pool(x).view(-1, 128))


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class Dataset(data.Dataset):
    def __init__(self, dataframe, list_IDs, target_type, outputs_a, pretrained=False):
        self.dataframe = dataframe
        self.list_IDs = list_IDs
        self.target_type = target_type
        self.outputs_a = outputs_a
        self.pretrained = pretrained

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        y = self.dataframe.target_class.get(ID)
        if self.target_type == 2:
            y = y.split(",")

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y]
        else:
            labels_ohe = np.zeros(self.outputs_a, dtype=np.float32)
            for idx, label in enumerate(range(self.outputs_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        file = DATA_DIR + self.dataframe.id.get(ID)
        img = cv2.imread(file)

        if img is None:
            raise ValueError(f"Could not read image: {file}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        img = img.astype(np.float32) / 255.0

        if self.pretrained:
            img = (img - self.mean) / self.std

        X = torch.from_numpy(img).permute(2, 0, 1).float()

        return X, y


# --------------------------------------------------
# Data loaders
# --------------------------------------------------
def read_data(target_type, outputs_a, pretrained=False):
    train_ids = list(xdf_train.index)
    val_ids = list(xdf_val.index)
    test_ids = list(xdf_test.index)

    train_set = Dataset(xdf_train, train_ids, target_type, outputs_a, pretrained=pretrained)
    val_set = Dataset(xdf_val, val_ids, target_type, outputs_a, pretrained=pretrained)
    test_set = Dataset(xdf_test, test_ids, target_type, outputs_a, pretrained=pretrained)

    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# Save model summary
# --------------------------------------------------
def save_model(model):
    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))


# --------------------------------------------------
# Build model
# --------------------------------------------------
def model_definition(outputs_a, pretrained=False):
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, outputs_a)
    else:
        model = CNN(outputs_a)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    save_model(model)

    return model, optimizer, criterion, scheduler


# --------------------------------------------------
# Metrics
# --------------------------------------------------
def metrics_func(metrics, aggregates, y_true, y_pred):
    def f1_score_metric(y_true, y_pred, avg_type):
        return f1_score(y_true, y_pred, average=avg_type, zero_division=0)

    def cohen_kappa_metric(y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred)

    def accuracy_metric(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def matthews_metric(y_true, y_pred):
        return matthews_corrcoef(y_true, y_pred)

    def hamming_metric(y_true, y_pred):
        return hamming_loss(y_true, y_pred)

    xsum = 0
    xcount = 0
    res_dict = {}

    for xm in metrics:
        if xm == 'f1_micro':
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            xmet = matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet = hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet
        xsum += xmet
        xcount += 1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcount > 0:
        res_dict['avg'] = xsum / xcount

    return res_dict


# --------------------------------------------------
# Process targets
# --------------------------------------------------
def process_target(target_type):
    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []

        if len(final_target) == 0:
            class_names = []
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join(str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    elif target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names = xtarget
        xdf_data['target_class'] = final_target

    else:
        class_names = []

    return class_names


# --------------------------------------------------
# Find best threshold on validation
# --------------------------------------------------
def find_best_threshold(y_true, probs, threshold_list):
    best_threshold = 0.5
    best_f1 = -1.0

    for thr in threshold_list:
        preds = (probs >= thr).astype(int)
        score = f1_score(y_true, preds, average='macro', zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = thr

    return best_threshold, best_f1


# --------------------------------------------------
# Main train + val + test
# --------------------------------------------------
def train_and_test(train_ds, val_ds, test_ds, list_of_metrics, list_of_agg, save_on, outputs_a, pretrained=False):
    model, optimizer, criterion, scheduler = model_definition(outputs_a, pretrained)

    best_val_metric = -1.0
    best_threshold = THRESHOLD

    for epoch in range(n_epoch):
        # ---------------- TRAIN ----------------
        model.train()
        train_loss = 0
        steps_train = 0
        train_logits_list = []
        train_targets_list = []

        with tqdm(total=len(train_ds), desc="Train Epoch {}".format(epoch + 1)) as pbar:
            for xdata, xtarget in train_ds:
                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()
                output = model(xdata)
                loss = criterion(output, xtarget)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                steps_train += 1

                train_logits_list.append(output.detach().cpu())
                train_targets_list.append(xtarget.detach().cpu())

                pbar.update(1)
                pbar.set_postfix_str("Train Loss: {:.5f}".format(train_loss / steps_train))

        train_logits = torch.cat(train_logits_list, dim=0)
        train_targets = torch.cat(train_targets_list, dim=0).numpy()
        train_probs = torch.sigmoid(train_logits).numpy()

        train_preds_default = (train_probs >= THRESHOLD).astype(int)
        train_metrics = metrics_func(list_of_metrics, list_of_agg, train_targets, train_preds_default)
        avg_train_loss = train_loss / steps_train

        # ---------------- VALIDATION ----------------
        model.eval()
        val_loss = 0
        steps_val = 0
        val_logits_list = []
        val_targets_list = []

        with torch.no_grad():
            with tqdm(total=len(val_ds), desc="Val Epoch {}".format(epoch + 1)) as pbar:
                for xdata, xtarget in val_ds:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    output = model(xdata)
                    loss = criterion(output, xtarget)

                    val_loss += loss.item()
                    steps_val += 1

                    val_logits_list.append(output.detach().cpu())
                    val_targets_list.append(xtarget.detach().cpu())

                    pbar.update(1)
                    pbar.set_postfix_str("Val Loss: {:.5f}".format(val_loss / steps_val))

        val_logits = torch.cat(val_logits_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0).numpy()
        val_probs = torch.sigmoid(val_logits).numpy()

        epoch_best_threshold, epoch_best_val_f1 = find_best_threshold(
            val_targets,
            val_probs,
            THRESHOLD_LIST
        )

        val_preds = (val_probs >= epoch_best_threshold).astype(int)
        val_metrics = metrics_func(list_of_metrics, list_of_agg, val_targets, val_preds)
        avg_val_loss = val_loss / steps_val

        # ---------------- TEST ----------------
        test_loss = 0
        steps_test = 0
        test_logits_list = []
        test_targets_list = []

        with torch.no_grad():
            with tqdm(total=len(test_ds), desc="Test Epoch {}".format(epoch + 1)) as pbar:
                for xdata, xtarget in test_ds:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    output = model(xdata)
                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    steps_test += 1

                    test_logits_list.append(output.detach().cpu())
                    test_targets_list.append(xtarget.detach().cpu())

                    pbar.update(1)
                    pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

        test_logits = torch.cat(test_logits_list, dim=0)
        test_targets = torch.cat(test_targets_list, dim=0).numpy()
        test_probs = torch.sigmoid(test_logits).numpy()

        test_preds = (test_probs >= epoch_best_threshold).astype(int)
        test_metrics = metrics_func(list_of_metrics, list_of_agg, test_targets, test_preds)
        avg_test_loss = test_loss / steps_test

        # ---------------- PRINT ----------------
        xstrres = "Epoch {}: TrainLoss {:.5f}".format(epoch + 1, avg_train_loss)
        for met, dat in train_metrics.items():
            xstrres += " Train {} {:.5f}".format(met, dat)

        xstrres += " - ValLoss {:.5f}".format(avg_val_loss)
        for met, dat in val_metrics.items():
            xstrres += " Val {} {:.5f}".format(met, dat)

        xstrres += " - TestLoss {:.5f}".format(avg_test_loss)
        for met, dat in test_metrics.items():
            xstrres += " Test {} {:.5f}".format(met, dat)

        xstrres += " - BestThr {:.2f}".format(epoch_best_threshold)
        print(xstrres)

        scheduler.step(epoch_best_val_f1)

        # ---------------- SAVE BEST MODEL ----------------
        if epoch_best_val_f1 > best_val_metric and SAVE_MODEL:
            torch.save(model.state_dict(), "model_{}.pt".format(NICKNAME))

            # save threshold to text file
            with open("threshold_{}.txt".format(NICKNAME), "w") as f:
                f.write(str(epoch_best_threshold))

            xdf_dset_results = xdf_test.copy()
            xfinal_pred_labels = []

            for i in range(len(test_preds)):
                joined_string = ",".join(str(int(e)) for e in test_preds[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results['results'] = xfinal_pred_labels
            xdf_dset_results.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

            print("Best model saved with validation F1-macro: {:.5f} and threshold: {:.2f}".format(
                epoch_best_val_f1, epoch_best_threshold
            ))

            best_val_metric = epoch_best_val_f1
            best_threshold = epoch_best_threshold

    print("Final best validation threshold:", best_threshold)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    xdf_data = pd.read_excel(FILE_NAME)

    class_names = process_target(target_type=2)

    # original competition split
    xdf_full_train = xdf_data[xdf_data["split"] == 'train'].copy()
    xdf_test = xdf_data[xdf_data["split"] == 'test'].copy()

    # one new change: split train into train + validation
    train_idx, val_idx = train_test_split(
        xdf_full_train.index.tolist(),
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    xdf_train = xdf_full_train.loc[train_idx].copy()
    xdf_val = xdf_full_train.loc[val_idx].copy()

    OUTPUTS_a = len(class_names)

    train_ds, val_ds, test_ds = read_data(
        target_type=2,
        outputs_a=OUTPUTS_a,
        pretrained=True
    )

    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    train_and_test(
        train_ds,
        val_ds,
        test_ds,
        list_of_metrics,
        list_of_agg,
        save_on='f1_macro',
        outputs_a=OUTPUTS_a,
        pretrained=True
    )