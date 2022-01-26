# tf.keras.Modelcheckpoint

[tf.keras.Modelcheckpoint](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint)

설명 : 모델 또는 모델의 Weights 저장을 위한 Callback <br>
fit과 결합해서 사용하여, 학습 중 특정 간격으로 모델 또는 가중치를 저장한다.<br> 
이후 저장된 가중치를 로드하여 저장된 상태에서 훈련을 계속할 수 있다.

___
```python
import tensorflow as tf

tf. keras.callbacks.ModelCheckpoint(
    filepath='../save', monitor='val_loss',verbose=0,save_best_only=False, save_weights_only=False,
    mode='auto', save_freq='epoch', options=None
)
```
## parameter
**moniter**  
- 모델 저장시 모니터링 할 메트릭스, 모델 저장 기준 <br>

**save_best_only** 
- True : moniter 되고있는 값을 기준으로 가장 좋은 모델/가중치 저장<br>
- False : 경우, 각 epoch 마다 저장

**save_weights_only** True의 겨우 모델의 weight 만 저장

**model** 
- max : accuracy와 같이 moniter 메트릭스 값이 클수록 좋을때
- min : loss와 moniter 메트릭스 값이 작을록 좋을때
- auto : 자동

**save_freq** 
- epoch : 매 에폭마다 저장
- 정수입력 : 숫자만큼의 배치를 진행하면 조델 저장

**options**
- tf.train.CheckpointOptions를 옵션으로 줄 수 있음<br>
분산환경에서 다른 디렉토리에 모델을 저장하고 싶을 경우 사용합니다.
