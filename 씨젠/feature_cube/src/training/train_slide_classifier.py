import os
import copy
import time
import pickle
import random
import torch
import torchvision
from datetime import datetime
from pathlib import Path

import src.training.train_config as cfg


def data_loader(data_path, diagnosis):
    data = []
    label = 0
    for condition in sorted(diagnosis):
        _path = os.path.join(data_path, condition)
        _files = list(Path(_path).glob("*.txt"))
        for file in _files:
            with open(file, 'rb') as f:
                features = pickle.load(f)
            data.append([features, label])

        if condition in ['M', 'U']:
            for file in _files:
                with open(file, 'rb') as f:
                    features = pickle.load(f)
                data.append([features, label])
        label += 1
    return data


def data_shuffle(data, batch_size):
    random.shuffle(data)

    result = [data[i * batch_size:(i + 1) * batch_size] for i in range((len(data) + batch_size - 1) // batch_size)]
    dataset = []

    for i in result:
        a, b = [], []
        for j in range(batch_size):
            try:
                a.append(i[j][0][0])
                b.append(i[j][1])
            except:
                pass
        dataset.append([a, b])
    return dataset


def train(model, num_epochs, batch_size, model_path, train_path, val_path, lr, diagnosis):
    since = time.time()
    _date = datetime.today().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(model_path, _date)

    # Path(model_path).mkdir(exist_ok=True, parents=True)
    Path(f"{model_path}/checkpoints").mkdir(exist_ok=True, parents=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[num_epochs / 4, (2 * num_epochs) / 4,
                                                                 (3 * num_epochs) / 4],
                                                     gamma=0.1)

    train_data = data_loader(train_path, diagnosis)
    val_data = data_loader(val_path, diagnosis)

    lowest_val_error = 1e10

    val_set = data_shuffle(val_data, int(batch_size / 16))

    best_model = copy.deepcopy(model)

    for epoch in range(num_epochs):
        train_set = data_shuffle(train_data, batch_size)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # Training
        model.train()

        running_corrects = 0.0
        running_loss = 0.0
        val_corrects = 0.0

        total_train_inputs = 0
        total_val_inputs = 0

        for inputs, labels in train_set:
            inputs = torch.tensor(inputs).float().cuda()
            labels = torch.tensor(labels).long().cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_corrects += torch.sum(preds == labels).double()
            running_loss += loss.item()
            total_train_inputs += len(inputs)

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"{model_path}/checkpoints/e{epoch}.pt")

        # Evaluation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_set:
                inputs = torch.tensor(inputs).float().cuda()
                labels = torch.tensor(labels).long().cuda()

                outputs = model(inputs)
                val_loss += criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_corrects += torch.sum(preds == labels).double()

                total_val_inputs += len(inputs)
        val_loss = val_loss / total_val_inputs
        if val_loss < lowest_val_error:
            lowest_val_error = val_loss
            torch.save(model.state_dict(), f"{model_path}/slide_classifier.pt")
            best_model = copy.deepcopy(model)

        time_elapsed = time.time() - since
        print(f"Epoch: {epoch + 1}/{num_epochs}, lr: {lr:.4f}, "
              f"train_loss: {running_loss / total_train_inputs:.4f}, "
              f"val acc: {val_corrects / total_val_inputs:.4f}. "
              f"Time {time_elapsed // 60:.0f}m {time_elapsed % 60:.4f}s")
        scheduler.step()

    time_elapsed = time.time() - since
    print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.4f}s")
    return best_model


def test(model, test_path, batch_size, diagnosis) -> None:
    since = time.time()

    test_data = data_loader(test_path, diagnosis)

    test_set = data_shuffle(test_data, int(batch_size / 16))

    # Evaluation
    model.eval()
    test_corrects = 0.0
    total_test_inputs = 0
    with torch.no_grad():
        for inputs, labels in test_set:
            inputs = torch.tensor(inputs).float().cuda()
            labels = torch.tensor(labels).long().cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_corrects += torch.sum(preds == labels).double()

            total_test_inputs += len(inputs)

    time_elapsed = time.time() - since
    print(f"Test acc: {test_corrects / total_test_inputs:.4f}. "
          f"Time {time_elapsed // 60:.0f}m {time_elapsed % 60:.4f}s")

    return


def train_classifier(
        diagnosis,
        model_path,
        data_path,
        batch_size,
        num_epochs,
        lr,
        dropout):
    train_path = os.path.join(data_path, 'train')
    val_path = os.path.join(data_path, 'val')

    model = torchvision.models.densenet201()
    model.classifier = torch.nn.Linear(model.classifier.in_features, 3)

    model._dropout = torch.nn.Dropout(dropout)

    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model).cuda()
    model.cuda()

    return train(
        model=model,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_path=model_path,
        train_path=train_path,
        val_path=val_path,
        lr=lr,
        diagnosis=diagnosis)


def test_classifier(
        model,
        diagnosis,
        data_path,
        batch_size):
    test_path = os.path.join(data_path, 'test')

    model.cuda()
    test(
        model=model,
        test_path=test_path,
        batch_size=batch_size,
        diagnosis=diagnosis)
    return


if __name__ == "__main__":
    _model = train_classifier(
        diagnosis=cfg.diagnosis,
        model_path=cfg.slide_classifier_saving_path,
        data_path=cfg.slide_classifier_data_path,
        batch_size=cfg.slide_classifier_batch_size,
        num_epochs=cfg.slide_classifier_num_epochs,
        lr=cfg.slide_classifier_lr,
        dropout=cfg.slide_classifier_dropout,
    )

    test_classifier(
        model=_model,
        diagnosis=cfg.diagnosis,
        data_path=cfg.slide_classifier_data_path,
        batch_size=cfg.slide_classifier_batch_size,
    )
