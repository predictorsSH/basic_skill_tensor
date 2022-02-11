# torch.nn 없이 신경망 만들기

[torch.nn](https://tutorials.pytorch.kr/beginner/nn_tutorial.html)

설명 : torch.nn 활용하지 않고 pytorch 연산만으로 모델을 만들기

## 데이터 셋 다운 
```python

from pathlib import Path
import requests
import pickle
import gzip

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/master/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")
    
from matplotlib import pyplot
import numpy as np

pyplot.imshow(x_train[0].reshape((28, 28)), cmap="gray")
```
이미지가 len*784 로 저장되어있다. 
그리고 numpy 배열이 아닌 torch.tenosr로 변환해야한다.

```python
import torch

x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape
```
## 신경망 만들기

pytorch는 랜덤 또는 0으로만 이루어진 텐서를 생성하는 메서드를 제공한다. <br>
해당 메서드로 weight과 bias를 생성할 수 있다. <br>
__requires_grad__ 메서드를 활용해 텐서의 매 계산시에 기울기(gradient)를 기록할 수 있다.

```python
import math

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()
bias = torch.zeros(10, requires_grad=True)
```
활성화 함수로 log softmax를 구현<br>
@는 matrix multiplication 연산을 나타냄<br>
forward pass 구현
```python
def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    return log_softmax(xb @ weights + bias)

bs = 64  # 배치 크기

xb = x_train[0:bs]  # x로부터 미니배치(mini-batch) 추출
preds = model(xb)  # 예측
preds[0], preds.shape
```
preds 텐서는 텐서 값 이외에도 grad_fn을 담고 있다. 이 함수를 역잔파를 위해 사용한다.

손실함수를 사용하기 위한 음의 로그 우도를 구현(negative log-likelihood) 하고  손실을 체크
```python
def nll(input, target):
return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0:bs]
print(loss_func(preds, yb))
```

정확도 구현
```python
def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

print(accuracy(preds, yb))
```

훈련 루프를 생성하여 훈련반복하여 weight과 bias를 업데이트 하게 되는데,<br>
이때, torch.no_grad() 내에서 실행한다. 이유는 업데이트시 진행되는 계산에서는 gradient가 기록되지 않아야 하기 때문이다.
```python
from IPython.core.debugger import set_trace

lr = 0.5  # 학습률(learning rate)
epochs = 2  # 훈련에 사용할 에폭(epoch) 수

for epoch in range(epochs):
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
마지막에 grad.zero_()를 통해서 기울기를 0으로 설정해야한다.
그렇지 않으면 매 루프마다 발생한 gradient를 계속해서 누적 집계하기 때문이다.

