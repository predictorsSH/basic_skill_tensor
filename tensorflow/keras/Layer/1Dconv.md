# 1D 합성곱 연산 눈으로 확인하기

tensorflow를 활용해 1D 합성곱 연산 과정 이해하기

---
필요 모듈 임포트
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv1D, Reshape, Flatten, MaxPool1D
```

테스트를 위한 데이터 생성 <br>
(1,5) shape의 데이터 이며, 쉽게 말해 5개의 피처를 가진 1개 행 데이터이다.
이 데이터 하나를 convolution 연생을 진행해보겠다.
```python
x = np.array([[1,2,3,4,5]], dtype= 'float64')
print(x.shape)
```

Reshape이 필요하다. (피처크기,1)로 reshape 해준다. 아래와 같이 Rehsape레이어를 활용하거나 
```python
x=Reshape((5,1))(x)
```
아래와 같이 numpy의 reshape 함수를 사용한다. 
```python
x.reshape(-1,5,1)
```
conv1d에 data를 input할때, (배치크기,피처크기,1) 이렇게 3차원으로 들어가야한다. <br> 
Reshape layer에는 배치크기를 생략해준 shape이 튜플 형태로 들어간다.

## 합성곱 연산

convolution layer를 생성하고, 연산을 수행해보자
```python
conv1d = Conv1D(filters=1,kernel_size=2,activation=None,use_bias=False)
x = conv1d(x)
```
연산 결과를 확인하기 쉽게 filters=1 kernel_size=2, activation=None, use_bias=False로 설정했다.
합성곱 연산에서 활성화 함수를 생략하고, + bias를 생략한 것이다.<br>
strides는 default가 1이다.

먼저, kernel를 확인해보자
```python
conv1d.kernel
```
커널이 다음과 같이 생성되었다
```
<tf.Variable 'conv1d_16/kernel:0' shape=(2, 1, 1) dtype=float32, numpy=
array([[[0.8088273 ]],

       [[0.01093018]]], dtype=float32)>
```
합성곱 연산 결과를 확인하자

```python
x
```
결과는 아래와 같다.
```
<tf.Tensor: shape=(1, 4, 1), dtype=float32, numpy=
array([[[0.83068764],
        [1.6504451 ],
        [2.4702024 ],
        [3.28996   ]]], dtype=float32)>
```
input data [1,2,3,4,5]에 커널 [0.8088273, 0.01093018]이 strides=1 씩 내려가면서 곱연산이 수행 된 것을 알수 있다.<br>
예를 들어, output의 0.83068764 = 1*0.8088273+2*0.01093018 이다.

## 풀링 

이제 pooling을 해보자.
output 결과를 stride=2, max pooling을 수행해 볼것이다.
예상되는 결과값은 [1.6504451,3.28996]이다.
```python
pool = MaxPool1D(pool_size=2)
pool_x = pool(x)
pool_x
```
결과는 아래와 같다. 예상과 일치한다.
```
<tf.Tensor: shape=(1, 2, 1), dtype=float32, numpy=
array([[[1.6504451],
        [3.28996  ]]], dtype=float32)>
```

## flatten
마지막으로 dense layer와 연결하기 위해서, 차원을 1차원으로 낮춰주는 과정이 필요하다.
Flatten 레이어를 활용한다.
```python
flatten = Flatten()(pool_x)
print(flatten)
```
결과는 아래와 같다!
```
tf.Tensor([[1.6504451 3.28996  ]], shape=(1, 2), dtype=float32)
```
## filters
필터 개수를 늘리면 어떻게 될까?
필터 개수를 2개로 합성곱 연산을 수행한 결과다. (커널 weight값 변동으로 위 결과와 일치하지 않음)

```
<tf.Tensor: shape=(1, 4, 2), dtype=float32, numpy=
array([[[ 1.3336239, -2.0821478],
        [ 2.0676873, -3.283961 ],
        [ 2.8017507, -4.485774 ],
        [ 3.535814 , -5.6875877]]], dtype=float32)>
```
필터 개수가 1개일 때는 출력결과가 (1,4,1)이었다. 필터 개수가 2일때는 (1,4,2)가 되었다!
마지막 차원이 필터의 차원이다. pooling 역시 마지막 차원이 2가 된다.

```
<tf.Tensor: shape=(1, 2, 2), dtype=float32, numpy=
array([[[ 2.0676873, -2.0821478],
        [ 3.535814 , -4.485774 ]]], dtype=float32)>
```

그렇다면 flatten을 수행하면 어떻게 될까?
```
tf.Tensor([[ 2.0676873 -2.0821478  3.535814  -4.485774 ]], shape=(1, 4), dtype=float32)
```
flatten은 필터개수가 하나이 던지 두개이던지 데이터 차원이 2차원이다.
무조건 일자로 쭉 펴준다는 이야기이다.
