import time
import torch
import torch.nn.functional as F
from pathlib import Path
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet

import train_config as cfg


def create_dir(_dir: str) -> None:
    Path(_dir).mkdir(exist_ok=True, parents=True)


def make_weights_for_balanced_classes(images, n_classes):
        count = [0] * n_classes
        weight = [0] * len(images)
        weight_per_class = [0.] * n_classes
        for item in images:
            count[item[1]] += 1
        N = float(sum(count))
        for i in range(n_classes):
            weight_per_class[i] = N / float(count[i])
        for idx, val in enumerate(images):
            weight[idx] = weight_per_class[val[1]]
        return weight


def get_data_loader(data_dir: str, batch_size: int):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.639, 0.462, 0.735], std=[0.235, 0.243, 0.151])]),
        'test': transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.639, 0.462, 0.735], std=[0.235, 0.243, 0.151])])
    }

    # Create datasets
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(data_dir + "/train/", data_transforms['train'])
    image_datasets['test'] = datasets.ImageFolder(data_dir + "/test/", data_transforms['test'])

    weights = make_weights_for_balanced_classes(image_datasets['train'].imgs, 
        len(image_datasets['train'].classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    loaders = dict()
    loaders['train'] = torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0)
    loaders['test'] = torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0)
    print("Data is loaded")
    return loaders


def train(model, loaders, num_epochs, model_save_dir, sl, lr):
    since = time.time()

    create_dir(model_save_dir)

    best_test_acc = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
        milestones=[num_epochs/4, 2*num_epochs/4, 3*num_epochs/4], 
        gamma=0.1)

    for epoch in range(0, num_epochs):

        if epoch == 2:
            break

        steps = 0
        running_loss_train = 0.0
        running_corrects_train = 0.0

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        print(f"Epoch: {(epoch + 1):3d}, lr: {lr:6f}")
        model.train()
        for inputs, labels in loaders['train']:

            inputs = inputs.cuda()

            # Setting the soft labels
            sl_labels = []
            if sl:
                for l in labels:
                    assert l in (0, 1, 2)
                    if l == 0:
                        sl_labels.append([0.7, 0.1, 0.1])
                    elif l == 1:
                        sl_labels.append([0.1, 0.7, 0.1])
                    else:
                        sl_labels.append([0.1, 0.1, 0.7])
            else:
                for l in labels:
                    assert l in (0, 1, 2)
                    if l == 0:
                        sl_labels.append([1., 0., 0.])
                    elif l == 1:
                        sl_labels.append([0., 1., 0.])
                    else:
                        sl_labels.append([0., 0., 1.])

            new_labels = torch.tensor(sl_labels).cuda()

            # Training
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.kl_div(F.log_softmax(outputs), new_labels)
            _, preds = torch.max(outputs, 1)
            running_loss_train += loss.item()
            running_corrects_train += torch.sum(preds == labels.cuda())
            loss.backward()
            optimizer.step()

            running_acc = torch.sum(preds == labels.cuda()).double() / len(labels)

            if steps % 1000 == 0:
                time_elapsed = time.time() - since
                print(f"Epoch: {epoch + 1}/{num_epochs}, steps: {steps}/{len(loaders['train'])}," \
                    f"loss: {loss.item() / len(labels):.4f}, acc: {running_acc.double():.4f}," \
                    f"time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.4f}s.")
            steps += 1

        # Validation
        model.eval()
        running_corrects = 0
        with torch.no_grad():
            for inputs, labels in loaders['test']:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                cs, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels)

        epoch_test_acc = running_corrects.double() / len(loaders['test'].dataset)

        # Summary
        epoch_loss_train = running_loss_train / len(loaders['train'].dataset)
        epoch_acc_train = running_corrects_train.double() / len(loaders['train'].dataset)

        print(f"Train loss: {epoch_loss_train:.4f}, train acc: {epoch_acc_train:.4f}")
        print(f"Val acc: {epoch_test_acc:.4f}")

        # Saving model
        torch.save(model.state_dict(), f"{model_save_dir}/e{epoch}_va{epoch_test_acc:.4f}.pt")

        scheduler.step()

    time_elapsed = time.time() - since
    print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.4f}s")


def train_classifier(model_architecture, data_dir, batch_size, num_epochs, model_save_dir, sl, lr):

    patch_model = EfficientNet.from_name(model_architecture)
    patch_model._fc = torch.nn.Linear(patch_model._fc.in_features, 3)

    if torch.cuda.device_count() > 1:
        patch_model = torch.nn.DataParallel(patch_model).cuda()

    patch_model.cuda()

    loaders = get_data_loader(data_dir, batch_size)

    train(
        model=patch_model,
        loaders=loaders,
        num_epochs=num_epochs,
        model_save_dir=model_save_dir,
        sl=sl,
        lr=lr)


if __name__ == "__main__":
    print("Patch classifier training...")
    train_classifier(
        model_architecture=cfg.patch_classifier_model_architecture,
        data_dir=cfg.data_dir,
        batch_size=cfg.patch_classifier_batch_size,
        num_epochs=cfg.patch_classifier_num_epochs,
        model_save_dir=cfg.patch_classifier_model_save_dir,
        sl=cfg.patch_classifier_sl,
        lr=cfg.patch_classifier_lr)
