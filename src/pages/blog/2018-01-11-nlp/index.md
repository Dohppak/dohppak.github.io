---
title: "자연어 처리에서 딥러닝의 역할"
date: "2019-01-11"
path: /blog/nlp
tags: DeepLearning, NLP
layout: post
---

본 문서는 조강현 교수님의 자연어처리를 위한 딥러닝 B세션의 강의를 정리한 문서입니다.

## NLP & Deep Learning

[Natural Language Processing (Almost) from Scratch](http://www.jmlr.org/papers/volume12/collobert11a/collobert11a.pdf)  에 따르면 딥러닝은 자연어 처리의 난제 중, 품사 태깅 (POS), 청킹 (CHUNK), 개체명 인식 (NER), 시맨틱 역할 레이블링 (SRL)을 푸는 역할을 수행하고 있다고한다. 하지만 학계를 살펴본다면 자연어 처리로 해결하고 하는 문제는 더 많은 상황이다. 현재 sota를 차지하는 논문들을 살펴본다면  Classification, Language model, Neural machine translation, Question answering, Machine reading comprension 등등, 다양한 분야의 언어 문제에서 딥러닝이 사용되고 있다. (예시: [paperswithcode](https://paperswithcode.com/area/nlp))

사실상 사람들이 사회에서 메세지를 주고받는, 커뮤니케이션의 되는 가장 근간은 언어다. 언어를 통해서 사람들이 소통하게 되고, 사회의 복잡성만큼이나 언어의 복잡성도 상당할것으로 보인다. 이러한 언어를 담고있는것이 바로 Text 문자이다. 문자를 통해 사람들이 사용하는 언어의 일부를 이해하고 그들이 문자를 통해 해결하고자 하는 문제들을 딥러닝이 도와주고 있다고 생각한다.

<img src='../img/Sota.png'>

# Data-Driven Algorithm Design

- Algorithm : 어떤 문제를 푸는데 있어서 필요한 일련의 명령입니다.
    - a Sequence of instructions that solves a problem.

- 머신러닝 : 예전에는 문제가 주어졌을 때, 해당 문제에 알맞는 알고리즘을 만드는 것이 일이였으나, 머신러닝에서는 문제를 결정하는 것이 어렵기 때문에 데이터로 부터 알고리즘을 학습시킵니다.
    - 데이터의 Sample이 있습니다
    - ML 모델이 문제를 풀도록 훈련을 시킵니다.
    
- Supervised Learning
    - Provide
        - Training Example : N개의 input과 output으로 구성된 샘플
        - Loss function : 모델의 추정 ouput과 실제 값인 y을 평가할 지표 
        $$
        L(M(x),y) > 0
        $$
        - Evaluation : 모형을 평가할 validation set과 test set으로 검증을 합니다.
        
    - Decide
        - Hypothesis Set : 모델의 아키텍쳐를 설정하는, 하이퍼 파라미터를 수정하게 만듭니다.
        - Optimization Algorithm : 각 모형들의 Performance인 loss function을 낮출 수 있는 머신을 학습합니다.
        
앞으로 우리가 정해야 하는 3가지 요소는
- 가설 집합 (Hypothesis set)
- 비용 함수 (Loss function)
- 최적화 알고리즘 (Optimization algorithm)
    

## Network Architecture?

- 비순환 그래프(DAG)라고 합니다.
    - 높은 수준의 추상화가 되어있어서 백앤드에 대한 고려를 적게 할 수 있습니다.
    - 객체지향 프로그래밍에 최적화 되어있습니다.
    - 코드의 재사용에 용이합니다.

## Hypothesis set in Deep learning

DAG를 만드는 행위입니다. 노드들을 만들고 노드들의 합인 아키텍처를 만드는 행위가 가설집합을 만드는것 입니다. 뉴럴넷에서 Hypothesis set을 결정함에 따라서, 다음과 같은 요소들이 고려되게 됩니다. 예를들어 layer의 갯수에 따라서 새로운 가설집합이 생기며, dropout이나 
- 네트워크 아키텍처가 정해지면, 하나의 가설 집합이 됩니다.
- 각기 다른 가중치 매개변수 값에 따라서 가설집한에 속한 모델이 달라집니다. 
- Solid Circle : 가중치 값이라고 생각합니다.
- Dash Circle : 백터화된 인풋값입니다.
- Squares : 연산입니다. 

<img src='../img/graph.png' width=70%>

## Loss function in Deep Learning
input이 주어졌을때, Y가 될수 있는 경우들의 확률 중 가장 높은 것은?

$$
f_{\theta}(x) = p(y'∣x)= ?
$$

여기서부터 Loss function이 시작된다. 주어진 데이터에 대해서 가장 가능성이 높은 Y Class를 찾고 싶기 때문이다.

#### Distribution of Y

그렇기 때문에 Y에 대한 확률분포에 대한 정보를 알아야합니다.
- 이진 분류 : 베르누이 분포 (Bernoulli)
- 다중 분류 : 카테고리 분포 (Categorical)
- 선형 회귀 : 가우시안 분포 (Gaussian)
- 다항 회귀 : 가우시안 믹스쳐 (Mixture of Gaussian)

분포에 대해서 하나의 X값을 넣으면 이제 확률을 리턴할 수 있습니다.

#### Probability

- 사건집합(event set) : 모든 가능한 사건의 집합(오메가 필드)
    - 이벤트 갯수가 유한일때 : 이산 (Descrete)
    - 이벤트 갯수가 무한일때 : 연속 (Continuous)
- 확률변수(Random Variable): 사건집합 안에 속하지만 정의되지 않은 어떤 값
- 확률(Probaility): 사건집합에 속한 확률 변수에게 어떤 값을 지정해 주는 함수입니다.
- 특성 (Properties)
    1. Non-negatives : 모든 확률은 비음수 입니다.
    2. Unit volume : 모든 확률의 합은 1입니다.
    
####  확률을 기반으로 비용함수를 정의해보자!
우리가 궁금한 확률은 X에 대한 조건부 Y의 확률입니다. $p(y'∣x)$ 인공 신경망 모델이 조건부 확률 분포를 출력하면, 이를 사용해서 비용함수를 정의 할 수 있습니다.
<br>

최대 우도 추정(Maximum Likelihood Estimation)
- 최대한 모델이 출력한 조건부 확률 분포가 훈련 샘플의 확률 분포랑 같도록 만드는게 목적입니다. 즉 모든 훈련 샘플이 나올 확률을 최대화 시켜야합나다.
$$
argmax_{\theta}logp_{\theta}(D) = argmax_{\theta}\sum_{n=1}^{N}logp_{\theta}(y_{n}|x_{x})
$$

최종적으로 비용함수는 음의 로그확률입니다.(Negative Log-porbabilities)

$$
L(\theta) = -\sum_{n=1}^{N}logp_{\theta}(y_{n}|x_{n})
$$

## Optimization in Deep Learning

Loss는 비순환 그래프(DAG)를 거쳐서 계산이 됩니다. 가설이 무수히 많기 때문에 모든 것을 다 시도해 보고 최적인 것을 고르기가 너무 어렵습니다.
- Local, Iterative Optimization : __Random Guided Search__
    - 장점 : 어떤 비용함수를 사용해도 무관
    - 단점 : 차원이 작을때는 잘 되지만, Sampling으로 인해서 시간이 오래걸린다는 단점이 있습니다.
    
- Gradient-based Optimization : __Gradient Descent__
    - 장점 : 탐색영역은 작지만 확실하게 방향을 잡을 수 있습니다.
    - 단점 : 학습률이 너무 크거나 작으면 최적값으로 못갈 수도 있습니다.

## Backpropagation

모델을 연결하는 비순환 그래프는 미분 가능한 함수로 구성되어 있습니다. 예를 들면 ouput이 새로운 input으로 들어가고, ouput으로 계산이 됩니다. 즉 chain 형태로 들어가 있습니다. 각 단계들을 미분값이 나오고, 다 곱셈을 적용하면 전체 미분 값을 찾을 수 있습니다.  즉 Loss function의 Gradient를 구할 수 있게되었습니다.

- 경사기반 최적화기법(Gradient-Based Optimization)
    - 파라미터의 수가 증가함에 따라서 시간이 오래걸립니다.
    - 훈련 샘플 전체의 Loss는 각 Loss의합으로 구해지며, 데이터가 많이 질 수록 오래 걸리게 됩니다.
- 확률적 경사 하강법(Stochastic Gradient descent)
    - 전체 비용은 훈련 샘플의 일부를 선택 후 나눠서 계산을 한 전체 비용의 근사값이다.
    - 왜 Stochastic 인가? 매번 계산을 할때마다, 골라지는 데이터셋에 따라서 조금씩 performance가 달라진다.
- Early Stopping
    - 과적합을 방지하기 위해, Validation Set의 loss가 가장 낮은 곳에서 훈련을 멈춥니다.
- 적응적 학습률(Adaptive Learning Rate)
    - 확률적 경사하강법은 Learning rate에 민감합니다. 이를 보안하기 위해서 적응형으로 다양한 학습법을 제안합니다.


```python

```
