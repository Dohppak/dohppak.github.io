---
title: "Recurrent Neural Network"
date: "2019-05-29"
path: /blog/rnn
tags: Sequence-Model ,RNN, DeepLearning
layout: post
---

## Sequence Model
Sequence 모델들의 특징인, 길의의 순서와 위치가 중요한 데이터를 input으로 받을때, 처리할 수 있는 모형입니다. 이러한 sequence model이 등장하게 된 이유는 이전의 feed forward 모형을 생각해보면 좋습니다. 한번 5개의 단어와 6개의 input 길이를 가지는 문장을 Input으로 받았다고 생각해봅시다. 그렇다면 뉴럴넷의 첫 input은 어떻게 설계 되어야할까요? 이럴때 문제가 발생하게 됩니다.

- 두개의 예시의 길이가 모두 같지 않게 된다는 문제점을 가지게 됩니다.
- 서로 다른 위치에서 학습된 feature는 공유되지 않습니다.

즉 위의 문제점을 보안하기 위해 Sequence 모델이 등자앟게 됩니다. 또한 두번째 문제점의 경우에는 예를들어 앞에서 등장한 데이터가 뒤에 영향을 주는 경우입니다. 예를들면 The 뒤에 명사가 오는 것이 예시가 될수 있겠죠? 이러한 연속적인 정보는 feedforward network로는 해석하기가 힘들게 됩니다.

### RNN
만약 우리가 문장을 좌에서 우로 읽는다고 가정해 봅시다. 처음 읽는 단어를 $x^{(1)}$ 이라고 합시다. 이 하나의 데이터만을 가지고 뉴럴넷을 통해서 output sequence를 예측하게 됩니다. 그 다음 단어 $x^{(2)}$가 input으로 들어온다면, 그때는 이전 sequence의 activation value를 반영하여 두번쨰 sequence의 activation을 연산하는데 사용이 됩니다. 즉 이전 시간의 정보를 활용하여, 이번 시간의 모형을 예측하는데 사용하는 것입니다. 정리한다면 RNN은 이전 sequence의 activation 값이 현재 sequence로 전이되어 학습하게 됩니다.

<img src="../img/RNN.png">

### Notation
일반적으로 데이터는, 총 데이터셋의 관점으로 몇번쨰 데이터인지를 결정하고, 그 다음은 단일 데이터의 sequence들은 Element의 길이를 가지게 됩니다. 각 sequence는 Time 축에 따라서 sequence를 형성하게 되므로 sequence속의 데이터를 표현하는 Notation이 존재하게 됩니다.

- $T_{x}$ : input 데이터 하나의 Sequence의 길이입니다.
- $T_{y}$ : output 데이터 하나의 Sequence의 길이입니다.
- $X^{(i)<t>}$ : $i$번째 트레이닝 데이터의 t번쨰 타임 스탭입니다.
- $[l]$ : $l^{th}$ 번쨰 레이어를 뜻합니다.

### RNN의 구조
<img src="../img/rnn_step_forward.png">

RNN의 기본구조를 먼저 살펴보려고 합니다. 위 그림을 보면 상당히 복잡헤 보입니다. 하지만 이것을 input과 output으로 나눠서 생각해 본다면 그렇게 어렵지 않게 해석 할 수 있습니다. Input을 먼저 봅시다. $a^{<t-1>}$ 이 맨 왼쪽에서 들어오고 있습니다. 그리고 아래에서는 $x^{<t>}$ 가 들어오고 있습니다. Output을 살펴봅시다. $a^{<t>}$가 오른쪽으로 나가고, 위로는 $\hat{y}^{<t>}$가 나가고 있습니다. 이제 앞으로 이 2개의 input과 2개의 output을 해석해 봅시다.

__Previous Hidden state : $a^{<t-1>}$__<br>
이것은 이전 타임스탭의 hidden state를 뜻합니다. 용어가 생소할수 있지만 흔히 우리가 feedforward network에서 보던 activation function을 지나쳐서 나온 값입니다.

__Current Input Sequence Element : $x^{<t>}$__<br>
이것은 현재 타입스탭의 input sequence element 입니다. 

__Current Hidden stata : $a^{<t>}$__<br>
이것은 현재 타임스탭의 hidden state를 뜻합니다. 두개의 input인 $a^{<t-1>}$, $x^{<t>}$에 각각의 Weight가 곱해진 뒤 더해집니다. 그리고 bias term을 더해준 후 tanh activation function을 지나게 됩니다. 이 현재의 hidden state는 다음번 time step으로 전달됩니다. 

$$
a^{<t>} = \text{tanh}(W_{ax}x^{<t>}+W_{aa}a^{<t-1>}+b_{a})
$$

위 식을 사용할때 좀더 간략하게 쓰기 위하여, input과 이전 step의 activation function을 horizental stack을 하는 방식이 있습니다. 예를들어 $a^{<t-1>}$가 100차원의 백터이고, $x^{<t>}$ 가 10000차원의 백터라고 가정해 봅시다. 이때 $a^{<t>}$가 100차원의 백터로 나와야한다면, 우리는 위 식을 다음과 같이 변환 시킬 수 있습니다. 덧셈을 이제 horizental stack 으로 변환시켜 버리는 것이죠. 
$$
a^{<t>} = \text{tanh}(W_{a}[x^{<t>},a^{<t-1>}]+b_{a})
$$

$$
W_{a} = [W_{ax}:W_{aa}]
$$
그리고 이떄 $W_{a}$의 차원은 (100,10100)이 됩니다.

__Current Output Sequence Element : $\hat{y}^{<t>}$__<br>
이것은 현재 타입스탭의 input sequence element 입니다. 이 설명 예제는 Sequence to Sequence를 예시입니다. Output Sequence Element는 Current hidden state에 Weight를 반영한다음에 bias term을 더해줍니다. 이때 current step의 hidden state는 이전 step의 정보를 담고 있게 됩니다. 때문에 시간의 변화에 따른 예측이 가능하게 되는 것입니다.

$$
\hat{y}^{<t>} = \text{softmax}(W_{ya}a^{<t>}+b_{y})
$$

이제 단일 element 에 적용되었던 RNN Cell을 전체 sequence에 적용한다면 다음과 같은 그림으로 Forward propagation을 설명할 수 있습니다.

<img src="../img/rnn-step.png">

### RNN Backpropagation
RNN의 Backprobagation은 살짝 복잡한듯 보이지만 하나씩 해석해본다면 이해할 수 있습니다.

__Loss__<br>
가장 먼저 계산되어야 하는 것은 바로 Loss값입니다. Loss 값은 예측된 $\hat{y}$와 실제 label인 $y$의 차이에 의해서 결정됩니다. Loss function은 일반적으로 Cross Entropy가 많이 사용됩니다.

__Gradient about $a^{<t>}$__<br>
그 다음 봐야하는 것은 바로 hidden state에 대한 변화량입니다. hidden state에 관여하는 값들은 Current Sequence의 input element 그리고 Previous hidden state 그리고 bias term입니다. 이때 input과 Previous hidden state는 $W_{aa}, W_{ax}$라는 Weight를 가지게 됩니다. 이들에 대해서 gradient를 구해야합니다. 이러한 구조로 인하여 Current step의 hidden derivative를 계산하게 된다면, 상당히 많은 Gradient를 구하는것이 쉬워집니다.

<img src="../img/rnn_cell_backprop.png">

위 수식들에서 반복되는 수식인 $\partial a^{<t>}$의 살펴본다면 다음과 같습니다.

$$
\partial a^{<t>} = (1-\text{tanh}(W_{ax}x^{<t>}+W_{aa}a^{<t-1>}+b)^{2})
$$

### RNN의 종류
<img src="../img/rnncate.png">

RNN의 종류는 총 4가지 경우가 있습니다. 이것은 input과 output sequence의 데이터가 어떠한 형식인지에 따라서 나눌수 있습니다. 

__One-to-One__<br>
이는 사실 단일 input과 단일 output이므로 feedforward network와 큰 차이가 없습니다. 

__One-to-Many__<br>
이는  Image Captioning 과 같은 Task 입니다. image을 input으로 받아서 이미지를 설명하는 단어의 sequence를 추출합니다. Many-to-One의 경우에는 Sentiment Classification과도 같은 task를 해결 할 수 있습니다. 예를 들어서 어떤 상품의 후기 문장에서 어떤 감정이 느껴지는지, 예의가 바른 문장을 구사하는지를 구별할수 있을 것입니다. sequence of words에서 sentiment을 분류하는 문제가 됩니다. 

__Many-to-Many__<br>
이 경우에는 2가지 경우가 있습니다. Time step의 텀이 있는 경우에는 보통 input 과 output의 시퀀스의 길이가 다르거나, input을 충분히 잘 고려해야하는 번역의 문제를 해결할 수 있습니다. Time step의 텀이 없는 경우에는 frame level에서 Video Classification 문제가 있게 될 것입니다.

### RNN Implementation
RNN의 구현은 다음과 같이 진행됩니다. 

- Single RNN cell
- RNN forward passway
- RNN Backpropagation


```python
import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

### 1. RNN Cell 구현

위에서 단일 Time step의 RNN cell을 본다면, 2개의 input과 ($x^{\langle t \rangle}$,$a^{\langle t-1 \rangle}$) 2개의 ouput ($a^{\langle t \rangle}$ , $y^{\langle t \rangle}$)를 나오게 됩니다.

1. hidden state를 구현해야합니다. $a^{\langle t \rangle} = \tanh(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$
2. Predict Output을 구현해야합니다. $\hat{y}^{\langle t \rangle} = \text{softmax} (W_{ya} a^{\langle t \rangle} + b_y)$.
3. Backpropagation을 위해서 $(a^{\langle t \rangle}, a^{\langle t-1 \rangle}, x^{\langle t \rangle}, parameters)$ 을 cache에 저장해야합니다.
4. Output과 cache를 Return해줍니다. $a^{\langle t \rangle}$ , $y^{\langle t \rangle}$

$m$ 개의 데이터 셋이 있다고 가정하고 input과 hidden state의 차원을 기술해 본다면 $x^{\langle t \rangle}$ 은 $(n_x,m)$ 차원을 가지게 됩니다. 또한 $a^{\langle t \rangle}$ 은 $(n_a,m)$의 차원을 가지게 됩니다.


```python
def rnn_cell_forward(xt, a_prev, parameters):
    
    # parameter의 dict에서 데이터를 호출합니다.
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # 1. hidden state를 구현해봅시다.
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # 2. Predict Output를 구현해봅니다.
    yt_pred = softmax(np.dot(Wya, a_next)+by)
    # 3. Cache에 저장합시다.
    cache = (a_next, a_prev, xt, parameters)
    
    return a_next, yt_pred, cache
```


```python
# 파리미터를 init한 다음에 계산을 한다면 다음과 같습니다.
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
```

### 2. RNN Forward Pass

실제 RNN 모델의 경우에는 단일한 time-step이 적용되는 경우는 거의 없습니다. 이번에는 RNN cell이 10개가 붙어있다고 생각해 봅시다.

__Arguments__<br>
- x -- 모든 time-step의 input 데이터입니다. shape은 (n_x, m, T_x)결정됩니다. 
- a0 -- 초기 hidden state입니다. hidden state의 갯수와, 데이터의 갯수로 shape이 결정됩니다.(n_a, m) 
- parameters -- python dictionary로 다음과 같은 정보가들어옵니다.
    - Waa -- hidden state에 대한 Weight matrix입니다.(n_a, n_a)
    - Wax -- input에 대한 Weight matrix입니다.(n_a, n_x)
    - Wya -- hidden state에서 output으로 가는 Weight matrix 입니다. (n_y, n_a)
    - ba -- hidden state에 대한 Bias입니다. (n_a, 1)
    - by -- output에 대한 Bias입니다 (n_y, 1)

__Returns__<br>
- a -- 모든 time step에 대한 hidden state 백터입니다. (n_a, m, T_x)
- y_pred -- 예측된 Output입니다. (n_y, m, T_x)
- caches -- Backprop에 필요한 Caches입니다. (list of caches, x)

__Task__<br>
1. $a$인 hidden state vector의 공간을 zero vector로 만들어 줍니다.
2. $a_0$ (initial hidden state)을 초기화 합니다.
3. Time step을 기반으로 for loop를 통해서 RNN cell 을 돌려줍니다. :
    - $a$ ($t^{th}$ position)를 계산합니다. 즉 이전 스탭에서 현재 스탭으로 업데이트하는 것이죠.
    - $a$ ($t^{th}$ position)를 캐시에 저장해 줍니다.
    - $y_pred$를 다시 업데이트 해줍니다.
    - 캐시를 저장합니다.
4. 마지막 step의 $a$, $y$와 caches를 저장해줍니다.


```python
def rnn_forward(x, a0, parameters):
    # caches라는 cache를 저장할 list를 선언합니다.
    caches = []
    
    # Dimension을 맞추기 위해 input sequence 가준으로 unpacking 해줍니다.
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape
    
    # a와 y를 초기화 합니다.
    a = np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))
    
    # a_next를 초기화 합니다.
    a_next = a0
    
    # time step을 돌면서 rnn cell을 작동 시킵니다.
    for t in range(T_x):
        # 1. hidden step을 계산해 줍니다.
        a_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], a_next, parameters)
        # 2. 새로운 hidden step을 a에 반영해 줍니다.
        a[:,:,t] = a_next
        # y의 값 역시 업데이트 해줍니다.
        y_pred[:,:,t] = yt_pred
        
        # 결과값을 저장해 줍니다.
        caches.append(cache)
        
    caches = (caches, x)
    return a, y_pred, caches
```


```python
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Waa = np.random.randn(5,5)
Wax = np.random.randn(5,3)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}

a, y_pred, caches = rnn_forward(x, a0, parameters)
```

### 3. RNN Backpropagation Cell

이제 단일한 RNN Backpropagation Cell을 구현해 봅시다.

1. 비선형 함수인 tanh의 미분식을 구현해 줍시다.
2. input sequence인 dxt, Wax 와 관련된 Loss의 gradient를 계산해줍니다.
3. previous step hidden state인 a_prev와, Waa와 관련된 Loss의 gradient를 계산해 줍니다.
4. 상수 b와 관련된 gradient를 계산해줍니다.
5. dictionary에 gradient들을 저장해줍니다.


```python
def rnn_cell_backward(da_next, cache):
    (a_next, a_prev, xt, parameters) = cache
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    # 1. 비선형 함수인 tanh의 미분식을 구해봅시다.
    dtanh = (1-a_next**2)*da_next
    
    # 2. dxt와 dWax를 구해봅시다.
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)
    
    # 3. da_prev와 dWaa를 구해봅시다
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)
    
    # 4. 상수 b의 gradient를 계산합니다.
    dba = np.sum(dtanh, 1, keepdims=True)
    
    # 5. Dic 에 저장해줍니다.
    gradients = {"dxt":dxt, "da_prev":da_prev, "dWax":dWax, "dWaa":Waa, "dba":dba}
    return gradients
```


```python
np.random.seed(1)
xt = np.random.randn(3,10)
a_prev = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
b = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

a_next, yt, cache = rnn_cell_forward(xt, a_prev, parameters)
```

### 3. RNN Backpropagation

이제 cell 단위의 backpropagation이 짜여졌습니다. 그렇다면 한번 time step에 따라서 진행되는 backpropagation을 짜봅시다. 이때 위에서 언급했듯이 반복되는 수식인 $\partial a^{<t>}$를 잘 활용해야합니다.

$$
\partial a^{<t>} = (1-\text{tanh}(W_{ax}x^{<t>}+W_{aa}a^{<t-1>}+b)^{2})
$$

__Arguments__<br>
- da : 모든 time step의 hidden state들을 조사합니다. Shape은 다음과 같겠군요. (n_a, m, T_x)
- caches : forward pass에서 전달되어서 오는 caches값입니다.

__Returns__<br>
- gradients -- python dictionary containing:
    - dx -- Input data의 gradient 입니다. numpy-array of shape (n_x, m, T_x)으로 결정됩니다.
    - da0 -- 초기 hidden state의 gradient입니다, numpy-array of shape (n_a, m)입니다.
    - dWax -- input's weight matrix의 gradient입니다, numpy-array of shape (n_a, n_x) 이 됩니다.
    - dWaa -- hidden state's weight matrix의 gradient입니다., numpy-array of shape (n_a, n_a)입니다.
    - dba -- bais의 gradient입니다. shape (n_a, 1)으로 결정됩니다.


```python
def rnn_backward(da, caches):
    # cahce를 unpacking해줍니다.
    (caches, x) = caches
    (a1,a0,x1,parameters) = caches[0]
    
    # Dimension의 고정시켜 줍니다.
    n_a, m, T_x = da.shape
    n_x, m = x1.shape
    
    # Gradient를 초기화 해줍니다.
    dx = np.zeros((n_x,m,T_x))
    dWax = np.zeros((n_a,n_x))
    dWaa = np.zeros((n_a,n_a))
    dba = np.zeros((n_a,1))
    da0 = np.zeros((n_a,m))
    da_prevt = np.zeros((n_a,m))
    
    # Time step을 기반으로 Loop를 돌려봅시다.
    for t in reversed(range(T_x)):
        gradients = rnn_cell_backward(da[:,:,t]+da_prevt,caches[t])
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        dx[:,:,t]=dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        
    da0 = da_prevt
    
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa,"dba": dba}
    
    return gradients
```


```python
np.random.seed(1)
x = np.random.randn(3,10,4)
a0 = np.random.randn(5,10)
Wax = np.random.randn(5,3)
Waa = np.random.randn(5,5)
Wya = np.random.randn(2,5)
ba = np.random.randn(5,1)
by = np.random.randn(2,1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
a, y, caches = rnn_forward(x, a0, parameters)
da = np.random.randn(5, 10, 4)
gradients = rnn_backward(da, caches)
```


```python
y.shape
```




    (2, 10, 4)




```python

```
