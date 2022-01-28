# DATASET 과 DATALOADER

설명 : 
- 파이토치는 torch.utils.data.Dataloader 와 torch.utils.data.Dataset 두가지 요소로 데이터셋을 컨트롤한다.
- Dataset은 샘플과 label을 저장하고, Dataloader는 Dataset을 순회 가능한 객체로 감싼다.

---

### FASHION MNIST 불러오기

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",  # 학습/테스트 데이터 저장 경로
    train=True,  #학습용
    download=True, #root에 데이터가 없는 경우 다운로드
    transform=ToTensor() #feature label transform 지정
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

```
- root : 학습/테스트 데이터가 저장되는 경로
- train : 학습용인지 테스트용인지 지정
- download : root에 데이터가 없을 때 다운로드
- transform : feature와 label의 변형을 지정

___

### 데이터 셋 순회

Dataset에 list 처럼 직접 접근 할 수 있다.
아래와 같이 training_data[index]를 사용하여, 시각화 할 수 있다.

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows+1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item() #랜덤하게 index 생성
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

```
---
### 사용자 정의 데이터셋

사용자 정의 데이터 셋은 반듯이 &#95;&#95;intit&#95;&#95; , &#95;&#95;len&#95;&#95;, &#95;&#95;getitem&#95;&#95; 
세가지를 구현해야 한다.

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

```

#### &#95;&#95;intit&#95;&#95; 

데이터 객체가 생성될때 한번만 실행된다. 이미지와 annotation_file이 포함된 디렉토리와 trasnform, target_trasnform을 초기화 한다.

#### &#95;&#95;len&#95;&#95;

데이터 샘플 개수를 반환한다.

#### &#95;&#95;getitem&#95;&#95;

주어진 인덱스(inx)에 해당하는 샘플을 데이터셋에서 불러오고 반환한다.<br>
인덱스를 기반으로 디스크에서 위치를 식별한다.<br>
텐서이미지와 라벨을 사전형으로 반환한다.

- read_img:이미지를 텐서로 변환


### DataLoader로 학습용 데이터 준비하기

Dataset의 feature와 label을 미니배치로 전달하고, 학습 epoch마다 데이터를 섞어서 과적합을 피하기위한<br>
추상화한 순회 가능한 객체

```python

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()  #squeeze : [1,28,28] -> [28,28]
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```