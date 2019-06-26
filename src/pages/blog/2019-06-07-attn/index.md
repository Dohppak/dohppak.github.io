---
title: "Attention Mechanism"
date: "2019-06-07"
path: /blog/attn
tags: Sequence-Model, Attention, DeepLearning
layout: post
---

본 게시물은 Attention Network에 대한 [lilianwengh](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)을 블로그 게시글을 번역한 글입니다. 또한 [distill](https://distill.pub/2016/augmented-rnns/#attentional-interfaces), [mchromiak](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XPoI49MzYWp)의 글을 참고하여 함께 정리했습니다.

### Attention mechanism

이번 포스팅에서는 Attention mechanism에 대해서 알아보려고 합니다. 우리는 일상생활에서 다양한 자극 속에서 목적에 맞는 중요한 단서에 집중해서 의사결정을 하게 됩니다. 그리고 반복을 통해 어떤 목적에 있어서 중요한 요소가 무엇인지 학습하게 됩니다. 예를 들어서 신호등 앞에서는, 시야중에 빨간색과 녹색을 찾기 위해서 집중하게 되는것과 비슷합니다. 아래 예시는 image caption에 관련된 예시입니다. 우리가 만약 caption에 줄이 쳐진 단어를 찾는 목적이 있다면, 해당 이미지에 집중하는 것과 비슷합니다.
<img src="../img/attend-tell.png">


이와 비슷하게 우리는 언어를 인식하는데 있어도 Attention mechanism을 사용합니다. 예를들어 She is eating a green apple이라는 문장이 있다고 해봅시다. 이 문장들 보고, 누가 그녀는 무엇을 '먹었어'? 라는 질문을 한다면, 우리는 apple을 찾아야합니다. 이렇듯 우리는 어떤 목적을 달성하기 위해서는 특정 정보에 주의를 기울여야합니다.
<img src="../img/xample-attention.png" width=70%>

이러한 Attention은 보면 다양한 정보들 중 어떤 정보에 주의를 기울이는가, 라고 생각할 수 있습니다. 이를 다르게 표현한다면 어떤 정보에 weight(가중치)를 주는 것인가 설명할 수 있습니다. 이 가중치를 학습을 통해 결정하는 것이 바로 Attention mechanism입니다.

### Seq2Seq Model의 한계

[lilianweng](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)의 블로그 에서는 Attention model의 시작부분을 seq2seq model의 한계로 부터 시작하게 됩니다. 이 seq2seq 모델은 Language Model Task를 해결하기 위해 등장하게 됩니다. 이 task는 input sequence를 target sequence로 변환시켜주는 형식의 many to many 방식의 sequence model입니다. 두 sequence 모두다 임의의 길이를 가지고 있습니다. 이것을 좁은 범위에서는 Language Model Task로 한정지을 수도 있지만 many-to-many분야의 다양한 domain과 task에 접목시키기도 가능합니다.

__seq2seq architecture__
- __Encoder__: input sequene의 정보를 압축해서 일정해 정보를 처리한 fixed length의 Context vector를 뽑는 과정입니다. 일반적으로 RNN 계열의 hidden state를 계산하는 과정이라고 생각하시면 됩니다.
- __Decoder__: 압축된 Encoder의 vector를 가지고, output를 생산하는 과정입니다. Encoder vector의 마지막 hidden state를 Decoder vector의 init input으로 사용하게 됩니다.
<img src="../img/encoder-decoder-example.png" width=70%>

이러한 fixed length의 Context vector는 한계를 가지게 됩니다. 점점 더 sequence의 길이가 길어질수록, 기억할 수 있는 한계가 존재하게 됩니다. 만약 번역의 문제의 경우에는, 어순의 차이로 인해서 점점 더 긴 문장을 기억해야하는 상황에서 이는 매우 중요한 문제로 여겨지게 됩니다.

### Translation Task
Attention은 neural machine translation을 위해서 탄생하게 되었습니다. 어순의 차이가 있는 언어간의 차이를 커버해야하기 때문에 점점 더 긴 단어를 기억해야하는 문제가 발생하게 됩니다. 이를 RNN계열 encoding 방식에서는 계속 마지막 hidden state까지 학습을 하면서 연산을 해야했습니다. 이러한 문제를 해결하는 것이 바로 attention 입니다. Attention은 input source 와 context vecotr 사이의 shortcuts들을 만들게 되었습니다. 이러한 shortcut connections은 output에 의해서 weight를 학습하게 됩니다.

이 encodding된 context vector에 의해서 source와 target의 alignment가 학습되기 때문에, 우리는 forgetting 에 대해서 걱정할 필요가 적어지게 됩니다. 이러한 context vector는 3가지 정보를 답게 됩니다.
- encoder hidden states;
- decoder hidden states;
- alignment between source and target.
<img src="../img/encoder-decoder-attention.png" width=70%>

### Network Notation

이 attention을 구현하기 위한 구조는 Bidirectional-LSTM Attention 모형으로 해보려고합니다. 해당 모형은 다음과 같습니다.

<img src="../img/attn_model.png" width=70%> <br>

이제 구현을 위해 Attention mechanism의 notation을 살펴봅시다. 우선 기본적인 가정은 다음과 같습니다. source sequence $x$가 들어왔을때, 우리가 출력해야하는 target sequence는 $y$가 됩니다. 이때 각각의 길이를 $n,m$이라고 해봅시다.
$$
x = [x_1, x_2, \dots, x_n]
$$
$$
y = [y_1, y_2, \dots, y_m]
$$

이제 이 input sequence가 [bidirectional LSTM](https://www.coursera.org/lecture/nlp-sequence-models/bidirectional-rnn-fyXnn) 를 통과하면서 encoding과정을 거치게 됩니다. $a_i$는 두개의 방향에 따라서 형성되게 됩니다. 이제 sequence의 양방향에서 encoding된 hidden state를 concatenate해서 대표적인 $a_i$를 만들어 내겠습니다. 

$$
\overrightarrow{a_t} = \overrightarrow{LSTM}(i_t, \overrightarrow{a_{t-1}})
$$
$$
\overleftarrow{a_t} = \overleftarrow{LSTM}(i_t, \overleftarrow{a_{t-1}})
$$
$$
a_t = [\overrightarrow{a_t},\overleftarrow{a_t}]
$$

이제 이 $a_t$를 가지고 Context vector를 만들어 봅시다. 우선 Context vector를 $c_t$라고 하겠습니다. 이 $c_t$는 input sequence들의 hidden state에 attention weight를 곱한 후 sum해준 값이 됩니다.
$$
c_t = \sum_{i=1}^n \alpha_{t,i} a_i
$$

그렇다면 여기서 attention weight는 어떻게 형성되며 계산해야할까요? 우리의 attention weight는 Source와 Target 사이의 관계를 포함하고 있다고 했습니다. 즉 $ \alpha_{t,i}$의 의미는 바로 source와 target과의 alignment라고 해석할수 있겠습니다.

$$
\alpha_{t,i} = \text{align}(y_t, x_i)
$$

잠시 멈추고 위로 올라가서 다시 그림을 본다면, Context vector가 형성된 뒤, 우리는 다시 bidirectional\_LSTM 네트워크가 존재하는것을 알 수 있습니다. 네 이 과정이 바로 decoding 과정입니다. decoding과정을 거친 후 output을 예측하기 시작할 것입니다. Decoder의 hidden state는 이전 step의 hidden state 그리고 이전의 output, 마지막으로 context vector가 함께 형성하게 됩니다. $s_t=f(s_{t-1}, y_{t-1}, c_t)$ 이 decoder의 hidden state는 매우 중요합니다. 왜냐하면 우리의 alignment function을 형성하는데 사용되기 떄문입니다. 

$$
\text{align}(y_t, x_i) = \frac{\exp(\text{score}(s_{t-1}, a_i))}{\sum_{i'=1}^n \exp(\text{score}(s_{t-1},a_{i'}))}
$$

<img src="../img/attn_mechanism.png" width = 70%>

Decoding의 hidden state가 이제 다시 내려와서 alignment function을 계산하는데 사용이 됩니다. Alignment function은 $\alpha_{t,i}$로 표기가 되며, input position i와 output position t가 얼마나 잘 매치되는가를 나타냅니다. Bahdanau의 논문에서는 alignment score가 **feed-forward network** 로 표현이 되었습니다. 따라서 score function은 다음과 같이 표현이 됩니다. 그리고 일반적으로 우리는 이 attention score를 Additive(\*) 라고 부릅니다.

$$
\text{score}(s_t,a_i) =v_a^\top \tanh(W_a[s_t;a_i])
$$

### Code Implementation

코드 구현을 위해서 LSTM의 간단한 모델인 GRU로 사용하려고 합니다. 코드 구현은 크게 2가지 클래스로 형성이 됩니다. 바로 Enconder, Attention + Decoder로 나눠서 있습니다. 이 2가지를 Pytorch로 구현을 해보려고 합니다. 

__Encoder__<br>
Neural Translation Task를 풀기 위해서는 일단 input sequence를 Embedding 차원으로 mapping 시켜야합니다. 그리고 이 Embedding의 차원을 정의하기위해서는 input size와 hidden size가 필요합니다. 그 다음 이 Embedding의 output을 gru 모듈에 넣어주어야합니다. 따라서 우리가 해야하는 것은 크게 2가지입니다.
- Embedding
- GRU

nn.Embedding은 simple lookup table을 만들어 줍니다. lookup table은 embedding을 fixed dictionary와 size를 저장한 객체를 형성해 냅니다. 이떄 필요한 파라미터는 Input size와 Hidden size입니다.

nn.GRU의 경우에는 간소화된 버전의 LSTM입니다. 자세한 내용은 이전 포스팅을 참고해주시고, output으로는 output(seq\_len, batch, num\_directions * hidden\_size)과 hidden(num\_layers * num\_directions, batch, hidden\_size)을 내보냅니다.

forward를 확인해봅시다. input data가 nn.Embedding에 들어간 다음에 torch.view를 통해서 (1,1,-1)로 들어가는 것을 볼수 있습니다. 이는 tensor의 shape을 바꾸어주는 것입니다. 남은 shape은 [1,1,input size x hidden size]가 됩니다. 이는 seq\_len, batch, input\_size를 뜻하며 GRU로 들어가게 됩니다. 


```python
import torch.nn as nn
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        
    def forward(self, input_data, hidden):
        embedded = self.embedding(input_data).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

__Attention Decoder__<br>
Decoder 의 경우에는 Context vector를 연산하고, Decoding 과 Output인 예측된 Target sequence를 return하게 됩니다.
- Attention weight 연산
- Attention weight 적용
- Decoding
- Output 생성

이떄 Attention의 연산부분 부터 시작해 봅시다. attention은 양방양에서 오는 encoding 들의 concatnate를 진행한 후, Linear 연산을 넣어주게 됩니다. 이때 concatnate를 도와주는 것이 바로 torch.cat입니다. 이 Linear 연산의 결과에 softmax를 걸어주면 이제 attention weight를 계산하는 부분이 끝납니다.
$$
\text{align}(y_t, x_i) = \frac{\exp(\text{score}(s_{t-1}, a_i))}{\sum_{i'=1}^n \exp(\text{score}(s_{t-1},a_{i'}))}
$$
$$
\text{score}(s_t,a_i) =W_a[s_t;a_i]
$$

그 다음은 attn\_weights와 encoder의 output를 결합하여 attention이 적용된 context vector를 생성하는 과정입니다. Context vector는 attention weight와 encoder의 결과를 활용하여 진행됩니다. 
$$
c_t = \sum_{i=1}^n \alpha_{t,i} a_i
$$

이 단계에서는 nn.bmm이 사용됩니다. bmm은 Batch matrix multiplication의 약자입니다. 동일한 batch size를 가지는 matrix들을 곱의 연산을 해줍니다. 이 결과 attn이 적용된 context vector가 나오게 됩니다. 이제 decoder의 input을 만들기 위해서, 이 attention을 input과 concat을 해주고, 다시 linear function에 넣어줍니다. 비선형 연산으로 relu를 거쳐서 이제 decoder에 넣어주고, softmax를 통해서 output을 예측하시면 됩니다. 결과로는 decoder의 hidden state와 output 그리고 attention weight를 받아보실수 있습니다.


```python
MAX_LENGTH = 10

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # additive Alignment score function
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, inputs, hidden, encoder_outputs):
        embedded = self.embedding(inputs).view(1, 1, -1)
        embedded = self.dropout(embedded)
        # align function을 통해 attention weight를 구하게 됩니다.
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        # Context vector를 구하는 과정
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        # 초기의 embedding과, Context vector를 concate합니다.
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        # feedforward network를 다시 적용
        output = self.attn_combine(output).unsqueeze(0)
        # 비선형 연산 적용
        output = F.relu(output)
        # decoder에 넣어줍니다.
        output, hidden = self.gru(output, hidden)
        # softmax를 통해 다시금 y_hat을 연산!
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights
```


```python

```
