import os
import time
import random
import pickle
import torch
import torchvision
import numpy as np
import pandas as pd
import seaborn as sn
from itertools import cycle
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix, accuracy_score
from efficientnet_pytorch import EfficientNet

import src.testing.test_config as cfg


def data_loader(data_path, diagnosis):
    data = []
    for idx, condition in enumerate(sorted(diagnosis)):
        _path = os.path.join(data_path, condition)
        _files = list(Path(_path).glob('*.txt'))
        for file in _files:
            with open(str(file), 'rb') as f:
                features = pickle.load(f)
            data.append([features, idx, Path(file).stem])
    return data


# def data_loader(data_path, diagnosis):
#     class_to_index = {k:v for v,k in enumerate(sorted(['D', 'M', 'N']))}
#     data = []
#     data_path = "temp/feature_cubes"
#     csv_file = pd.read_csv("stomach_slide_distribution.csv")
#     folders = [str(f) for f in Path(data_path).glob('*') if f.is_dir()]
#     for folder in folders:
#         print(folder)
#         label = csv_file.loc[csv_file['file']==Path(folder).stem]['label'].tolist()[0]

#         label = class_to_index[label]
#         with open(f"{folder}/temp.txt", 'rb') as f:
#             features = pickle.load(f)
#         data.append([features, label, Path(folder).stem])
#     return data


def plot_confusion_matrix(true_labels, pred_labels):
    
    # idx_to_class = {k: v for k, v in enumerate(sorted(cfg.diagnosis))}
    cm = confusion_matrix(true_labels, pred_labels)
    df_cm = pd.DataFrame(cm, index=[f"True {i}" for i in sorted(cfg.diagnosis)], 
        columns=[f"Pred {i}" for i in sorted(cfg.diagnosis)])
    print(df_cm)
    # plt.figure(figsize=(10,7))
    return


def plot_roc_curve(true_labels, pred_labels, accuracy, n_classes=3):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_labels[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr['micro'], tpr['micro'], _ = roc_curve(true_labels.ravel(), pred_labels.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    lw = 2
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    colors = cycle(["aqua", "darkorange", "cornflowerblue"])
    idx_to_class = {k: v for k, v in enumerate(sorted(cfg.diagnosis))}
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label="ROC curve of class {0} (area = {1:0.2f})".format(idx_to_class[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC for {cfg.anatomy} data. Accuracy: {accuracy*100:.2f}%")
    plt.legend(loc="lower right")
    plt.show()
    return


def get_slide_model(slide_model_path: str):
    """
    Load slide-level model
    """
    model_state_dict = torch.load(slide_model_path)
    is_data_parallel = np.array(["module" in k for k in model_state_dict.keys()]).all()

    model = torchvision.models.densenet201()
    model.classifier = torch.nn.Linear(model.classifier.in_features, 3)
    # model = EfficientNet.from_name("efficientnet-b3", in_channels=3)
    # model._fc = torch.nn.Linear(model._fc.in_features, 3)
    if is_data_parallel:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(model_state_dict)

    model.cuda()
    return model


def test(model, test_path, diagnosis) -> None:
    since = time.time()

    test_set = data_loader(test_path, diagnosis)

    # Evaluation
    model.eval()
    test_corrects = 0
    total_test_inputs = 0
    true_labels, pred_labels = [], []
    all_labels, all_preds = [], []
    to_save = []
    with torch.no_grad():
        for inputs, labels, file in tqdm(test_set, total=len(test_set)):

            inputs = torch.tensor(inputs).float().cuda()
            labels = torch.tensor(labels).long().cuda()
            one_code_label = [0, 0, 0]
            one_code_label[labels.item()] = 1

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_corrects += int(preds.item() == labels.item())

            total_test_inputs += len(inputs)

            all_labels.append(labels.item())
            all_preds.append(preds.item())
            to_save.append([file, sorted(cfg.diagnosis)[labels.item()], sorted(cfg.diagnosis)[preds.item()]])


            true_labels.append(one_code_label)
            pred_labels.append(outputs.cpu().detach().tolist()[0])

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    time_elapsed = time.time() - since

    pd.DataFrame(to_save, columns=["file", "label", "pred"]).to_csv("from_test.csv", index=False)

    accuracy = accuracy_score(all_labels, all_preds)

    print(f"Test acc: {accuracy:.4f}. Time {time_elapsed // 60:.0f}m {time_elapsed % 60:.4f}s")

    print(f"\n{cfg.anatomy} confusion matrix:")
    plot_confusion_matrix(all_labels, all_preds)

    plot_roc_curve(true_labels, pred_labels, accuracy)

    return


def test_classifier(
        model,
        diagnosis,
        data_path):
    test_path = os.path.join(data_path, 'test')

    model.cuda()
    test(
        model=model,
        test_path=test_path,
        diagnosis=diagnosis)
    return


if __name__ == "__main__":
    _model = get_slide_model(cfg.slide_model_path)
    
    test_classifier(
        model=_model,
        diagnosis=cfg.diagnosis,
        data_path=cfg.slide_classifier_data_path
    )
