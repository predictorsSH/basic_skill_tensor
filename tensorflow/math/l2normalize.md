# tf.math.l2_normalize

[tensorflow api_docs](https://www.tensorflow.org/api_docs/python/tf/math/l2_normalize)

설명 : L2 norm을 사용한 정규화<br>

norm은 선형대수학에서 벡터의 크기를 측정하는 방법을 의미한다. L1, L2 norm이 있다. <br>
l2_normalize는 L2 norm을 사용해서 구한 벡터의 크기를 가지고 데이터를 정규화하는 방법이다.

___
## L2 Norm

![L2](C:\Users\user\IdeaProjects\basic_skill_tensor\tensorflow\math\img\l2_norm.PNG)

각 성분의 제곱의 합을 루트로 씌워준 값으로, 유클리디안 norm이라고도 부른다.

___

## tf.math.l2_normalize

설명 : axis 기준으로 아래와 같은 식으로 정규화 수행<br>

output = x/sqrt(max(sum(x**2), epsilon)) <br>

```python
import tensorflow as tf
arr = tf.constant([[1.0,1.0],[1.0,2.0]]) #예시 배열 

tf.math.l2_normalize( x=arr , axis=None, epsilon=1e-12, name=None, dim=None
)
```


예시 코드 :

```python
import tensorflow as tf

arr = tf.constant([[1.0,1.0],[1.0,2.0]])
print(tf.math.l2_normalize(arr, axis=0).numpy()) 
#[[0.7071, 0.4472],[0.7071,0.8944]]

#0.7071 = 1/sqrt(sum(1**2,1**2))
#0.4472 = 1/sqrt(sum(1**2,2**2))
#0.8944 = 2/sqrt(sum(1**2,2**2))


```
