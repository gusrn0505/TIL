import pickle
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from torchvision.utils import save_image
from src.utils.utils_torch import load_patch_loader


def add_pred_to_base(base_feature, predict, overlap, tile_size):
    patch_pred = []
    for path, pred, pred_label, confidence in predict:
        # patch 정보에서 col, row 정보를 추출
        # col, row 는 상하좌우 몇 번째 patch인지 알려줌. 
        path = Path(path).stem
        path = path.split('-')[1]
        path = path.split('.')[0]
        col = path.split('_')[0] 
        row = path.split('_')[1]

        # put in prediction info at the base feature-cube
        # base feature : (1, DMN, lsize, csize)
        # Feature cube의 각 칸마다 NDM의 SOFTmax 값을 대입함. 
        for i in range(3):
            base_feature[0][i][int(col)][int(row)] = pred[i]

        # it is for a heatmap
        # 이거 - 하는게 맞나? 이거 연산에서 실수가 생길 수가 있겠는걸. 
        # 왜 -1 이 되어있을까? 
        # 실제 Patch 좌표로 변환 및 label, s-core 추가하기. 
        patch_pred.append([(int(row) - 1) * (tile_size - (overlap * 2)),
                           (int(col) - 1) * (tile_size - (overlap * 2)),
                           pred_label, confidence])

    return base_feature, patch_pred


def base_feature_cube(lsize: int, csize: int):
    # lsize : length of the feature cubew. 256
    # csize : channels of the feature cube. 128 => 이거 그냥 RGB가 아닌 것 같은데 폭을 말하는 듯?
    a = torch.tensor([0.])
    k = torch.tensor([0.])
    for i in range(0, lsize * csize):
        k = torch.cat([k, a], dim=0) # [0,0] 의 값을 위아래로 쌓기 (lsize*csize, 2)
    r = np.reshape(k[1:], (1, lsize, csize))
    # Q. 왜 이렇게 r을 만든걸까? 

    a = torch.tensor([0.])
    k = torch.tensor([0.])
    for i in range(0, lsize * csize):
        k = torch.cat([k, a], dim=0)
    g = np.reshape(k[1:], (1, lsize, csize))

    a = torch.tensor([0.])
    k = torch.tensor([0.])
    for i in range(0, lsize * csize):
        k = torch.cat([k, a], dim=0)
    b = np.reshape(k[1:], (1, lsize, csize))

    base = torch.cat([r, g, b], dim=0)

    return np.reshape(base, (1, 3, lsize, csize))


def predict_patch_label(patch_model, data_path: str, batch_size: int):
    """
    Patch-level predictions
    """

    # load dataset
    tile_loader = load_patch_loader(data_path=data_path, batch_size=batch_size)

    # patch-level classification
    patch_level_pred = []
    patch_model.eval()
    with torch.no_grad():
        for data in tile_loader:
            x, path = data
            x = x.cuda()
            pred_outputs = patch_model(x)
            # 확률로 변환
            pred_outputs = F.softmax(pred_outputs, dim=1) # NDM 각 Class에 속할 확률 값. 

            #score와 label을 모두 제공하는 듯. 
            # prediction은 0,1, 2(N)  값으로 제공 
            #Q. torch.max의 값이 confidence, prediction 값 둘다 저장할 수 있나? 
            # confidence는 softmax 값 중에서 가장 큰 값만 보여주는 것인듯? 
            confidence, predictions = torch.max(pred_outputs, 1)

            for i, pred in enumerate(predictions):
                # Normal patches are considered as nothing
                if pred != 2:
                    patch_level_pred.append([
                        path[i],
                        pred_outputs.cpu().tolist()[i],
                        predictions.cpu().tolist()[i],
                        confidence.cpu().tolist()[i]
                    ])
    return patch_level_pred


def build_feature_cube(
        patch_model,
        data_path: str,
        lsize: int,
        csize: int,
        overlap: int,
        batch_size: int,
        tile_size: int):

    base_feature = base_feature_cube(lsize, csize) #(1, 3, lsize, csize) 구조. tensor로 0의 값으로 차 있다. 
    patch_level_pred = predict_patch_label(
        patch_model=patch_model,
        data_path=data_path,
        batch_size=batch_size)

    # slide_feature : feature cube 값 채우기 
    # patch_prediction : [각 Patch 별 x,y 좌표, confidence, label 값] append 
    slide_feature, patch_prediction = add_pred_to_base(
        base_feature=base_feature,
        predict=patch_level_pred,
        overlap=overlap,
        tile_size=tile_size)

    return patch_prediction, slide_feature

# feature cube 저장하기. 
def save_feature_cubes(slide_features, slide_name: str, feature_cube_path: str):
    Path(f"{feature_cube_path}/{slide_name}").mkdir(exist_ok=True, parents=True)
    img_path = f"{feature_cube_path}/{slide_name}/temp.jpg"
    save_image(slide_features[0], img_path)
    txt_file = f"{feature_cube_path}/{slide_name}/temp.txt"
    with open(txt_file, "wb") as f:
        pickle.dump(slide_features.tolist(), f)
    return
