import torch
from tqdm import tqdm
from torchvision import transforms, models, datasets


def main():

    transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = datasets.ImageFolder("./test_tiles/test/", transformation)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0)

    patch_classifier_path = "./lossdiff/merged_colon_tiles_lossdiff_balanced.pkl"
    model = models.densenet201()
    model.classifier = torch.nn.Linear(model.classifier.in_features, 3)
    model.load_state_dict(torch.load(patch_classifier_path))
    model.cuda()


    model.eval()
    test_corrects = 0.0
    total_inputs = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, total=len(loader.dataset)):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            test_corrects += torch.sum(preds == labels).double()
            total_inputs += len(inputs)

    print(f"Test acc: {test_corrects / total_inputs:.4f}")

    return


if __name__ == "__main__":
    main()
