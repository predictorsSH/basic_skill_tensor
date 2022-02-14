# torch.nn 활용하여 신경망 만들기(without_torchnn refactoring)

[torch.nn](https://tutorials.pytorch.kr/beginner/nn_tutorial.html)



## 신경망 만들기
nn.Module 하위 클래스를 만들어, 모델 리팩토링

__refactoring 전__
```python

#weight, bias
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)

#활성화함수, foward 계산 
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

bs = 64  # 배치 크기

xb = x_train[0:bs]  # x로부터 미니배치(mini-batch) 추출
preds = model(xb)  # 예측


```

__refactoring 후__

```python
from torch import nn
import torch 
import math

#활성화 함수, 손실함수 refactoring, 
import torch.nn.functional as F

loss_func = F.cross_entropy #log softmax와 로그우도 손실함수를 결합하는 단일 함수. 모델에서 log softmax 제거 


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

model = Mnist_Logistic() # 모델 인스턴스화
```
---

model.parameters() 및 model.zero_grad()활용하여  훈련루프 리팩토링

__refactoring 전__

```python
from IPython.core.debugger import set_trace

lr = 0.5  # 학습률(learning rate)
epochs = 2  # 훈련에 사용할 에폭(epoch) 수

for epoch in range(epochs):
    #n=데이터 개수, bs는 배치크기
    for i in range((n - 1) // bs + 1): 
        #         set_trace()
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
```

__refactoring 후__
```python
def fit():
    for epoch in range(epochs):
        #n=데이터 개수, bs는 배치크기
        for i in range((n - 1) // bs + 1):
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()
```

optim을 이용하여 리팩토링 하기<br>
각 매개변수를 수동으로 업데이트 하는 대신, 옵티마이저(optimizer)의 step 메소드를 사용하여 업데이트를 진행

```python
from torch import optim

def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)

model, opt = get_model()

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))

```

Dataset, DataLoader를 이용하여 리팩토링<br>
데이텉 미니배치를 자동으로 생성

```python

from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs)

model, opt = get_model()

for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

print(loss_func(model(xb), yb))
```

검증추가하기<br>
<br>
훈련데이터를 shuffle 하는 것은 배치와 과적합사이의 상관관계를 방지하기 위해 중요. (데이터의 편향 제거)<br>
검증 데이터셋에 대해서는 역전파(backpropagation)가 필요하지 않으므로 메모리를 덜 사용. 따라서 배치크기 키움.
```python
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

model, opt = get_model()

for epoch in range(epochs):
    model.train()#학습시에 train() 호출!!
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()#예측시에 eval() 호출!!
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))
```
학습과, 검증데이터 셋 각각 손실을 계산하는 프로세스가 있으므로, 하나의 배치에 대한 손실을 계산하는 자체함수로 만들기.
```python

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
```
get_data 함수를 활용하여 학습 및 검증 데이터 셋에 대한 dataloader 출력,
get_data와 fit()함수를 활용하여 모델 학습 및 손실 검증
```python

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)

```