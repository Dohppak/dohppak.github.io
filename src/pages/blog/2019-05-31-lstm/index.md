---
title: "Long Short Term Memory Networks (LSTM)"
date: "2019-05-31"
path: /blog/lstm
tags: Sequence-Model, LSTM, DeepLearning
layout: post
---


본 게시물은 Long Short Term Memory Networks에 대한 [colah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)을 블로그 게시글을 번역한 글입니다. 매우 잘 정리되어있는 게시글이고 많은 걸 배웠습니다. 또한 코드구현은 Andrew ug 교수님의 [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome) 강의를 참고했음을 밝힙니다.

## LSTM (Long Short Term Memory Networks)

### RNN의 한계
RNN은 좋은 Sequence 모델이지만 몇가지 문제가 존재합니다. 예를들어 문장의 길이가 길어짐에 따라서 주어와 후반부에 등장하는 동사의 형이나 시제를 일치해야하는 문제를 들어봅시다. RNN 모듈은 이전 hidden state를 잘 반영하는 모델이지만, hidden state가 많아 질수록 기억해야하는 정보량 역시 많아지는 문제가 발생합니다. 이를 우리는 Long Term dependency라고 부르는데요. 또한 feedforward network에서 발생했던 Vanishing gradients의 문제와 Exploding gradients문제를 가지고 있음을 알 수 있습니다.

예시를 통해 좀더 Long Term dependency에 대해서 알아봅시다. 만약 I grew up in France..... I speck fluent $h_{t+1}$ 이라는 문장이 있다고해봅시다. 과연 $h_{t+1}$에는 어떤 단어가 들어가면 적절할까요?. 우리는 French인 것을 알 수 있습니다. French의 결정적인 단서인 France가 될 것입니다. 하지만 RNN의 경우에는 France라는 인풋이 너무 앞쪽의 Time-step에 있기 때문에 쉽게 비중있는 정보를 반영하기 힘들 것입니다.
<img src="../img/longtermdependencies.png">

### LSTM Networks
이와 같은 문제를 해결하는 LSTM네트워크는 1997년도에 Hochreiter & Schmidhuber에 의해서 제안되었습니다. LSTM의 특징은 Cell state 입니다. RNN에서 input과 이전의 hidden state 뿐만 아니라, 하나의 state가 추가된 것이죠. 저는 이 Cell state를 Long term memory를 모델링해서 생긴 정보라고 생각이 듭니다. 즉 input은 새로들어온 정보, 그리고 previous hidden state는 short term memory 그리고, Cell state는 long term memory라고 생각이 됩니다.

이렇게 3개의 정보들은 내부에서 4개의 게이트를 통과하게 됩니다. 더 정확하게는 input과 이전 step의 hidden state입니다. Gate는 말 그대로 정보의 흐름을 통제하는 문과 같습니다. 이 LSTM에서는 4개의 뉴럴넷이 그 역할을 하게 됩니다. 하나의 예시로, Sigmoid 함수의 Range는 0과 1사이입니다. 이 결과를 곱한다는 것이 의미하는 것은 정보를 어느정도로 반영할 것인가를 경정하는 것과 같습니다. 위 내용을 정리해서 LSTM의 특징을 말한다면 다음과 같습니다. 

- 3개의 information flow, 2개의 recurrent flows (Cell, hidden)
- Four neural network layers in one module

<img src="../img/fulllstm.png">
<img src="../img/lstmnotation.png">

노테이션을 확인해보면, 4개의 게이트는 뉴럴넷으로 표현되어있습니다. 핑크색 원은 pointwise operation입니다. 화살표들이 모이는 부분은 정보의 Concatenate 그리고 화살표들의 분리는 정보의 Copy가 발생한다고 생각하시면 됩니다.

### Step-by-Step LSTM Walk Through

__Forget Gate__<br>
Lstm의 정보의 흐름을 이해하기 위해서, 4개의 gate중 가장 가까이에 있는 Forget gate를 먼저 확인해 봅시다. 이는 $x_{t}$와 $h_{t-1}$를 input으로 받는 sigmoid layer인데요. Sigmoid의 Range는 위에서 말한것 처럼 0과 1사이입니다. 즉 Cell state의 특정 time-step의 정보를 기억할 것인지 잊어버릴것인지를 물어보는 단계인 것입니다.
<img src="../img/forget.png">

__Input Gate(Update Gate) & Tanh layer__<br>
두번째는 현재의 time step의 정보를 얼마나 cell state에 저장할 건지 결정하는 Gate가 있습니다. 이것은 2개의 부분으로 이루어 지는데요. 첫번째는 input gate layer입니다. forget gate와 모델링이 정확하게 일치합니다. 이전 step의 hidden step 과 현재의 input을 받아서 sigmoid로 조절하게 되는 것이죠. 하지만 input gate의 역할은 update입니다. 지금의 정보를 저장하는데 집중하는 것이죠. 두번째 부분은 바로 Tanh layer입니다. 여기서는 $tilda{C_{t}}$를 만들어 냅니다. 여기서는 tanh값이 사용됩니다. 그 결과 우리는 -1과 1사이의 range를 가지는 값을 얻게 됩니다. 이 둘의 * 연산은 새로운 Candidate value로 이후 Cell state를 얼마나 변화시킬 것인지 결정하게 됩니다. 
<img src="../img/input.png">

__Updata Cell state__<br>
Cell state의 update는 forget gate와 Input gate 그리고 Tanh layer의 의해서 이루어집니다. 하지만 하나는 곱의 연산 하나는 덧셈의 연산이 있습니다. 이 두가지 연산이 update를 하는 방식을 다르게 하고, Gate의 이름을 결정지었습니다. forget gate의 곱 연산은 이전 cell state를 어느정도로 잊어버릴것인가를 결정하게 됩니다. 반면 덧셈의 연산은 얼마난 cell state를 업데이트 할 것인지를 결정하게 되는 것이죠.
<img src="../img/cellupdate.png">

__Output gate__<br>
이제 output을 연산할 차례입니다. Output gate는 Sigmoid layer를 가지는 뉴럴넷과 Tanh를 거쳐서 오는 Cell state의 결합으로 형성됩니다. 현재의 정보가 반영된 Cell state에 한 번더 현재 정보를 기반으로 filter 를 거친다는 의미를 가지게 됩니다.
<img src="../img/output.png">

이렇게 LSTM의 흐름을 살펴보았습니다 매우 복잡할것 같은 수식도 사실은 4개의 뉴럴넷에 덧셈과 곱셈의 연산이 포함되면서, 형성된다는 것을 알 수 있었습니다. 각 gate가 하는 역할과 의미에 중점을 두고 파악하신다면 금방 이해가 될 것으로 예상이 됩니다.

### LSTM 코드 구현
추후 업데이트 됩니다.

### LSTM의 변형
colah의 블로그에서는 생각보다 재밌는 변형적인 LSTM역시 소개하고 있습니다. 

__peep-hole connection(핍홀)__<br>
첫번째는 Gers & Schmidhuber의 핍홀 커넥션입니다. 아주 간단하게는 forget, input, output gate에 각각 cell state를 반영해 주는 것입니다. Cell state를 Long term memory라고 생각한다면, 이제 모형에서 Long term memory를 조금더 많이 반영하여 연산을 진행하겠다는 소리가 될 것입니다.
<img src="../img/peephole.png">

__Couple forget and input gate__<br>
Cell state의 새로운 정보를 잊어버리고 저장하는 것에 대해서, 두가지의 결정을 동시에 하겠다는 의미입니다. 아래 그림을 보면  input gate가 없어지면서 (1-forget gate)가 input gate로 변한 것을 알 수 있습니다. 이는 잊어버리지 않는 것은 곧 기억된다는 것입니다. 이분법적으로 기억할 것과 잊을 것을 나누겠다는 의미로 생각이 됩니다.
<img src="../img/tied.png">

__GRU__<br>
드디어 조강현 교수님의 Gated Recurrent Unit이 등장했습니다. GRU의 경우에는 forget gate와 input gate를 update gate로 합쳐 주셨습니다. 또한 Cell state와 hidden state를 합치게 됩니다. 이로 인하여 구조적 특성을 반영하면서 좀더 간단한 LSTM 모델이 탄생하게 됬습니다. 수식을 본다면 위의 Couple forget and input gate의 구조가 반영되었으며, cell state와 hidden state의 결합으로, hidden state가 모든 gate 연산에 포함되었습니다. 단기기억을 매우 신중하게 고민하는 듯한 연상이 됩니다.
<img src="../img/GRU.png">

이번 포스트에서는 LSTM과 LSTM들의 변형을 공부하였습니다. 실제로 Andrew ug 교수님에 의하면 GRU와 LSTM 모형은 현재까지 많이 사용된다고 합니다. 무엇이 더 좋은 모형이냐는 사용 목적과 Task에 따라서 달라진다고 하니 좋은 GPU를 가지고 실험을 해보는게 가장 좋은 방식인 것 같습니다. 이외에도 앞선 Time step의 정보를 반영하는 Bi-directional RNN의 형태도 있으며, RNN의 레이어를 쌓아서 이전 레이어에 hidden step과, 동일한 레이어의 이전 time step의 hidden layer를 받아서 활성화를 형성하는 Deep RNN의 방식도 존재합니다. Sequence Model 은 Convolution Model만큼 DeepLearning의 중요한 요소임이 분명하므로 자세히 살펴보는것을 권장드립니다.


```python

```
