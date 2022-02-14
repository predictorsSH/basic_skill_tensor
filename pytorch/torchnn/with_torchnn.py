from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np
import math
from torch import nn
from torch import optim



def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


loss_func = F.cross_entropy

class Mnist_Logistic(nn.Module):

    # def __init__(self):
    #     super().__init__()
    #     self.weights = nn.Parameter(torch.randn(784,10)/ math.sqrt(784))
    #     self.bias = nn.Parameter(torch.zeros(10))

    # def forward(self, xb):
    #     return xb @ self.weights + self.bias

    #nn.Linear 이용
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):

        model.train() #학습시에 train() 호출!!
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval() #예측시에 eval() 호출!!
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)



if __name__ == '__main__' :

    #데이터 셋 다운
    DATA_PATH = Path("data")
    PATH = DATA_PATH / "mnist"

    PATH.mkdir(parents=True, exist_ok=True)

    URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
    FILENAME = "mnist.pkl.gz"

    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

    #데이터가 python 전용 포맷 pickle을 이용하여 저장되어있음, numpy 배열 포맷임
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

    pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
    pyplot.show()
    print(x_train.shape) #이미지가 len*784 로 저장되어있음 784=28*28 형태로 변환이 필요

    #데이터 변환
    x_train, y_train, x_valid, y_valid = map(
        torch.tensor, (x_train, y_train, x_valid, y_valid)
    )
    n, c = x_train.shape

    bs = 64
    epochs = 5
    lr=0.5

    #데이터셋 생성 및 학습
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    model, opt = get_model()
    fit(epochs, model, loss_func, opt, train_dl, valid_dl)
