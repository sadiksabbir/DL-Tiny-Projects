from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import cv2
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torchvision import models
from tqdm import tqdm
import os
import argparse

'''
MATCHING TEST CODE
Uses saved threshold from training.
Inference only.
'''

# --------------------------------------------------
# Arguments
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)
parser.add_argument("--split", default=False, type=str, required=True)

args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split

# --------------------------------------------------
# Settings
# --------------------------------------------------
BATCH_SIZE = 30
IMAGE_SIZE = 224
CHANNELS = 3

NICKNAME = "Callisto"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5


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
# Data loader
# --------------------------------------------------
def read_data(target_type, outputs_a, pretrained=False):
    test_ids = list(xdf_dset_test.index)
    test_set = Dataset(xdf_dset_test, test_ids, target_type, outputs_a, pretrained=pretrained)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)
    return test_loader


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
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, outputs_a)
    else:
        model = CNN(outputs_a)

    model.load_state_dict(torch.load('model_{}.pt'.format(NICKNAME), map_location=device))
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    save_model(model)

    return model, criterion


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
    if target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names = xtarget
        xdf_data['target_class'] = final_target

    elif target_type == 2:
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
    else:
        class_names = []

    return class_names


# --------------------------------------------------
# Test
# --------------------------------------------------
def test_model(test_ds, list_of_metrics, list_of_agg, outputs_a, pretrained=False):
    model, criterion = model_definition(outputs_a, pretrained)

    # load threshold saved during training
    threshold_file = "threshold_{}.txt".format(NICKNAME)
    if os.path.exists(threshold_file):
        with open(threshold_file, "r") as f:
            best_threshold = float(f.read().strip())
    else:
        best_threshold = THRESHOLD

    model.eval()

    test_loss = 0
    steps_test = 0
    test_logits_list = []
    test_targets_list = []

    with torch.no_grad():
        with tqdm(total=len(test_ds), desc="Testing the Model") as pbar:
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
    pred_labels = (test_probs >= best_threshold).astype(int)

    test_metrics = metrics_func(list_of_metrics, list_of_agg, test_targets, pred_labels)
    avg_test_loss = test_loss / steps_test

    xstrres = "TestLoss {:.5f}".format(avg_test_loss)
    for met, dat in test_metrics.items():
        xstrres += " Test {} {:.5f}".format(met, dat)
    xstrres += " Threshold {:.2f}".format(best_threshold)
    print(xstrres)

    xfinal_pred_labels = []
    for i in range(len(pred_labels)):
        joined_string = ",".join(str(int(e)) for e in pred_labels[i])
        xfinal_pred_labels.append(joined_string)

    xdf_dset_test['results'] = xfinal_pred_labels
    xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    xdf_data = pd.read_excel(FILE_NAME)

    class_names = process_target(target_type=2)

    xdf_dset_test = xdf_data[xdf_data["split"] == SPLIT].copy()

    OUTPUTS_a = len(class_names)

    test_ds = read_data(
        target_type=2,
        outputs_a=OUTPUTS_a,
        pretrained=True
    )

    list_of_metrics = ['f1_macro']
    list_of_agg = ['avg']

    test_model(
        test_ds,
        list_of_metrics,
        list_of_agg,
        outputs_a=OUTPUTS_a,
        pretrained=True
    )