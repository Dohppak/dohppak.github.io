---
title: "Word & Sentence Representation"
date: "2018-01-12"
path: /blog/word
tags: Representation, Embedding, NLP
layout: post
---

## Word Representation

우리가 사용하는 언어는 상당히 Abstract합니다. 언어는 grounding이 없다고 생각하시면 됩니다.
- Sentence : 다양한 길이를 가지는 토큰들의 Sequence입니다.
- Token : Vocabulary에 들어가는 요소입니다.
    - 형태소 단위
    - 어절 단위
    - 비트 숫자
    - 공백 단위

#### Encoding : 컴퓨터가 어떻게 Token을 인식하게 할까?

컴퓨터에게 단어를 숫자로 표현하기 위해서, 중복되지 않는 단어장을 만들고, 중복이 없는 index를 부여합니다. 하지만 관계없는 숫자로 인코딩하는것을 원하지 않습니다. 우리는 단어간의 관계를 만들고자 합니다.

- Onehot-Encoding : 모든 단어는 동등하게 독립적이다.
    - 총 길이가 결정된 단어장 벡터에서, 단어의 index에 위치하는 값은 1, 아닌 나머지는 0으로 구성합니다.
    - 모든 토큰간의 거리가 같습니다. 하지만 이 말은 모든 단어간의 의미공간에서 각 단어가 동등하다는 의미를 가집니다.

- Word Embedding (단어 임베딩)
    - 각 토큰을 연속 벡터 공간에 투영하는 방법입니다. 
    - Look-up Table : 각 one hot encoding 된 토큰에게 백터를 부여하는 과정입니다. 실직적으로는 one hot encoding 벡터에 벡터 공간을 내적하는 것입니다.
    - Word embedding 은 단어의 의미론적 공간입니다. 효율적으로 단어의 상대적인 의미공간을 만들어 낼 수 있습니다.
    - 이러한 embedding을 이용하여, POS 태깅, parse trees등을 만들어 낼 수 있습니다.
    
- Character Embedding (문자 임베딩)
    - 단어 임베딩은 문법적, 의미적 정보를 잡아낼 수 있다. 그러나 품사태깅(POS-tagging)이나 개체명 인식(NER)같은 task에서는 단어 내부의 형태, 정보 또한 중요하다.
    - 문자 임베딩은 미등재 단어(the unknown word)이슈에 대처 가능하다.
    - 중국

#### Word embedding (Word2vec)

Word2Vec 은 비슷한 문맥을 지니는 단어를 비슷한 벡터로 표현하는 distributed word representation의 방법입니다. 

#### Softmax regression
Word2vec은 softmax regression 입니다. softmax regression은 클래스가 3개 이상인 경우, 일반화된 logistic regression이라고 생각하면좋습니다. Logistic은 백터공간의 점 $x$가 주어졌을 때, 클래스가 $y$일 확률을 학습합니다. exponential의 range는 (0,+$\infty$)이기 때문에, $\frac{1}{1+\exp(\theta^{T}x)}$는 (0,1)의 범위를 지니게 됩니다.


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
```




    <torch._C.Generator at 0x11a540050>




```python
word_to_ix = {"hello":0, "world":1}
# 2개의 vocab에 있는 단어를, 5차원에 embedding 하자.
embed = nn.Embedding(2,5)
```


```python
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embed(lookup_tensor)
print(hello_embed)
```

    tensor([[ 0.6614,  0.2669,  0.0617,  0.6213, -0.4519]], grad_fn=<EmbeddingBackward>)


## Sentence Representation
내가 푸는 문제에 대해서 가장 적합한 representation이 무엇인가?

### CBOW
- Continuous Bag-of-word (CBOW)
    - 단어장을 단어 주머니로 보게되고, 이에 따라 단어의 순서를 무시합니다.
    - CBOW 모델은 크게 input, projection, output layer로 구성이 되어 있습니다. 
    - Generalize to bag of n-gram
        - Token의 N-gram을 적용하기도 편합니다.
    - Baseline 모델로 많이 사용이 됩니다.
    - 공간에서 가까우면 비슷한 의미, 아니면 다른 의미공간입니다.
    - 문장에 대한 표현은 단어 백터들을 평균시킨 벡터로 구합니다.
        - 하나의 operator node로 문장이 표현이 됩니다.

조금더 정확하게 말한다면, CBOW모델은 주어진 context word를 가지고, target word를 예측하는 것입니다. Language modeling과의 차이점은, CBOW는 Sequential 하지않고 (순서 무시), 그리고 확률적이지 않습니다. CBOW은 일반적으로 빠른 word embedding을 지원하고, 첫 initalize 하는데 도움이 많이 됩니다. 수식으로 이해를 해봅시다. Target word를 $w_{i}$라고 하고, $N$개의 context window가 있다고 할때, Context word는 $w_{i-1}, \dots, w_{i-N}$와  $w_{i-1}, \dots, w_{i-N}$가 됩니다.

$$
-\log p(w_i | C) = -\log \text{Softmax}(A(\sum_{w \in C} q_w) + b)
$$
$q_{w}$는 단어 $w$의 임베딩입니다. 


```python
window = 2
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
```


```python
vocab = set(raw_text)
vocab_size = len(vocab)
word_to_ix = {word: i for i, word in enumerate(vocab)}
```


```python
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i-2], raw_text[i-1],
              raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))
data[:5]
```




    [(['We', 'are', 'to', 'study'], 'about'),
     (['are', 'about', 'study', 'the'], 'to'),
     (['about', 'to', 'the', 'idea'], 'study'),
     (['to', 'study', 'idea', 'of'], 'the'),
     (['study', 'the', 'of', 'a'], 'idea')]




```python
data[0][0]
```




    ['We', 'are', 'to', 'study']




```python
class CBOW(nn.Module):
    def __init__(self):
        pass
    def forward(self,inputs):
        pass

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

make_context_vector(data[0][0], word_to_ix)
```




    tensor([ 3,  2, 22, 35])



<img src='../img/CBoW.png' width=70%>

### Skip-gram
- Relation Network(Skip-Bigram)
    - 문장이 주어졌을 때, 모든 단어들의 pair를 생각합니다. Pair의 representation을 찾게 됩니다.
    - 모든 관계를 고려하면서 복잡하게 딥러닝을 보게 됩니다. pair-word representation을 합니다.
    $$
    h_{t} = f(x_{t},x_{1}) + ... f(x_{t},x_{t-1})+ ... f(x_{t},x_{T})
    $$

<img src='../img/RN.png' width=70%>

### CNN

- Convolutional Networks in NLP
    - 로컬한 Structure가 중요하다면, 좁은 지역간의 단어의 관계를 잘 표현합니다.
    - K-gram을 계층적으로 불 수 있게 됩니다.
    - Convolution은 tokens - Multi-word - Phrases - Sentence를 볼수 있게 해준다.
    - 1D Convolution Network입니다.
    - 긴거리의 Dependency가 있다면 문제가 생깁니다. 특정한 window기준으로 보기 때문입니다.
    $$
    h_{t} = f(x_{t},x_{t-k}) + ... f(x_{t},x_{t})+ ... f(x_{t},x_{t+k})
    $$

<img src='../img/CNN.png' width=70%>

### Attention
- Self-Attention
    - CNN은 너무 local하게 보는것이고, Relation Network는 너무 global하게 보는것 아닌가?
    - CNN이 만약 가중치가 부여된 RN으로 본다면? window는 1, 나머지는 0 매우 hard한 가중치 이지만, 이를 soft하게 만든다면?
    $$
    h_{t} = \sum_{t`=1}^{T} \alpha(x_{t},x_{t'}) f(x_{t},x_{t'}) \\
    \alpha(x_{t},x_{t'}) = \frac {exp(\beta(x_{t},x_{t'}))}{\sum_{t`=1}^{T} exp(\beta(x_{t},x_{t'}))}
    $$
    - Long range & Short range dependency를 극복가능합니다.
    - 관계가 낮은 토큰은 억제하고, 관계가 높은 토큰은 강조 가능합니다.
    - 계산 복잡도가 높고 counting 같은 특정 연산이 어렵습니다. $O(T^2)$

<img src='../img/Self-Attention.png' width=70%>

### RNN

- RNN
    - Sentence 를 linear 한 time에 compression이 가능합니다.
        - 문장의 정보를 시간의 순서에 따라 압축 가능합니다.
    - 메모리를 가지고 있어서 현재까지 읽는 정보를 저장 할 수 있습니다.
    - Sequence 한 구조이기 때문에, 하나씩 읽어야해서 살짝 속도가 느리는 문제가 있다.
    - LSTM 과 GRU로 확장이 가능하다.
        - bidirectional network를 쓰게 됩니다.
