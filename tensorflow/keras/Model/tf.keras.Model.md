# tf.keras.Model

[tesnorflow api_dos](https://www.tensorflow.org/api_docs/python/tf/keras/Model#used-in-the-notebooks_1)

설명 : Model은 각 레이어를 그룹화한다. Model을 인스턴스화 하는 방법은 두가지가 있다. <br> 

1. Functional API Model 

- 시퀀스 모델에 비해 더 유연하게 모델을 생성할 수 있는 함수형(functional) API <br>
  
    - 예시 코드:

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(3,)) #input layer
dense = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs) #dense layer
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(dense) #output layer
model = tf.keras.Model(input=inputs, outputs= outputs) # Model, layer group
```

- 중간 tensors를 활용해 새로운 함수형 API 모델을 만들 수 있다. 모델의 특정 부분(sub_model)을 추출 할 수 있다.<br>
- 예시 코드:
```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(None,None,3))
processed = tf.keras.layers.experimental.preprocessing.RandomCrop(width=32, height=32)(inputs)
conv = tf.keras.layers.Conv2D(filters=2, kernel_size=3)(processed)
pooling = tf.keras.layers.GlobalMaxPooling2D()(conv)
feature = tf.keras.layers.Dense(10)(pooling)

full_model = tf.keras.Model(inputs,feature)
backbone = tf.keras.Model(processed,conv) #sub model
activation = tf.keras.Model(conv,feature)
```

2. subclassing the Model class
- init 에서 레이어 정의, call에서 모델의 전진 패스를 구현 <br>
- 예시 코드:
  
```python
import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(4)
        self.dense2 = tf.keras.layers.Dense(5)
    
    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)
    
model = MyModel
```

