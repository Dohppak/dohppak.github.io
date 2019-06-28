---
title: "딥러닝의 직관적 이해"
date: "2019-03-01"
path: /blog/deep
tags: beginner, DeepLearning
layout: post
---
본 포스팅은, 딥러닝을 시작하기 앞서서 만나게 될 생소한 단어들에 대한 기본적인 설명이나, 딥러닝이 기존의 전통적인 머신러닝 방법론들과의 차이를 보이기 위해서 작성하게 되었습니다. 포스팅은 Stanford university의 CS231n 과 [ratsgo님의 블로그](https://ratsgo.github.io/deep%20learning/2017/09/24/loss/)를 참고했습니다.

## Deep Learning이란?

일반적으로 우리가 알고 있는 딥러닝이란, 컴퓨터가 데이터로 부터 학습을 통해, 개념들 간의 관계와 hierarchy를 만들어서 복잡한 패턴을 이해하는 것입니다. 패턴을 이해하는 것으로 부터 우리는 어떤 문제를 풀 수 있을까요? 예를 들어서 다음과 같은 데이터가 주어졌다고 해봅시다. 
```python
2,4,6,8,10,? 
```
?에 들어갈 숫자는 무엇일까요? 우리는 아마 빠르게 ?가 12라고 알 수 있습니다. 2씩 증가하는 수열임을 눈치챘기 때문이죠. 데이터에는 내재하는 패턴이 존재합니다. 데이터 상에 존재하는 패턴을 통해서 예측이나 분류와 같은 과제를 수행하는 것입니다.

다시 돌아가서 딥러닝이란 무엇일까요? 일반적으로는 End to end feature learning 으로 Domain knowledge에 대한 의존성이 낮으면서 위의 Task를 해결할 수 있는 모형으로 알려져 있습니다.

<img src="../img/deeplearning.png">

- Traditional Pattern Recognition
    - 여러가지 Fixed 되어있고 Handcrafted 한 Feature Extreaction 기법을 통해서 Feature를 추출하고 학습을 시켰습니다. 이 당시에는 domain knowledge가 매우 중요했습니다.
    
- Mainstram Modern pattern Recognition
    - Feature Extraction 이후에 Unsupervised learning을 통해 mid-level feature가 뽑혀서 나오기 시작했습니다. (PCA)
    
- Representation are hierarchical and trained
    - Deep learning의 시대가 되었습니다. 각 layer는 feature를 learning 시키는 역할을 하게 되었습니다. 이는 인간이 feature를 이해하는 학습 시스템과 점점 더 비슷하게 변하게 됩니다.


### Data-Driven Algorithm Design
딥러닝에 들어가기전에 간단하게 몇가지 용어를 정리해봅시다. Learning이 중요해진 이유는 알고리즘의 기본적인 발상을 뒤집어 버렸기 떄문입니다. 간단하게 알고리즘, 머신러닝, Supervised Learning에 대해서 알아봅시다.

- Algorithm : 어떤 문제를 푸는데 있어서 필요한 일련의 명령입니다.
    - a Sequence of instructions that solves a problem.

- Machine Learning : 예전에는 문제가 주어졌을 때, 해당 문제에 알맞는 알고리즘을 만드는 것이 일이였으나, 머신러닝에서는 문제를 결정하는 것이 어렵기 때문에 데이터로 부터 알고리즘을 학습시킵니다.
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

### Deep Learning 구조
딥러닝은 모형들의 output이 다시 input으로 들어가서 학습을 이어나가는 Layer 구조를 가지는 모형입니다. 모형의 특징을 결정하는데는 다음과 같은 요소들이 필요하게 됩니다. 바로 모델을 구축하는 과정과, 그 다음은 모델을 학습하는 과정입니다. 앞으로는 밑의 개념들을 하나씩 정리해 보려고합니다. 

- 모델 Building
    - Connectivity patterns
    - Nonlinearity Modules
    - Loss function
- 모델 학습
    - Optimization
    - Hyper Parameters

### Connectivity Pattern
딥러닝은 여러 레이어를 쌓아나가는 구조입니다. 뉴런은 각 레이어에 있으며, 레이어들간의 연결관계에 따라서 패턴이 나눠집니다. 따라서 우리는 연결관계의 연결 패턴이 매우 중요하게 됩니다. 연결 패턴은 다음과 같이 존재합니다.
- Fully-Conntected
- Convolutional
- Dilated
- Recurrent
- Skip / Residual
- Random

### Nonlinearity modules
일반적인 뉴럴네트워크는 선형 연산후 비선형 함수를 사용하게 됩니다. 그렇다면 왜 비선형 함수를 사용하게 될까요? 이유는 input units들의 interaction을 확인하기 위해서 입니다. 만약 linear Mutiplication만 진행한다면, 결국 layer를 여러개 쌓을 이유가 없습니다. (Multiplication of linear transformation = another linear transformations.) 이 non-linear function으로 인해서 우리는 선형연산의 한계인 input units들의 interactions을 반영하지 못한다는 문제를 해결 할 수 있습니다.
- ReLU
- Sigmoid
- Tanh
- GRU
- LSTM

### Loss
손실을 정의하는 방식입니다.

- Cross Entropy
- Adversarial
- Variational
- MLE
- L1 and L2
- Reinforce

### Optimizer
학습의 진행을 도와줍니다.

- SGD
- Momentum
- RMSProp
- Adagrad
- Adam
- Second Order

### Hyper Parameter
- Learning rate
- Weight decay
- Layer size
- Batch size
- Dropout rate
- Weight initalization
- Data augmentation
- Gradient Clipping
- Momentum

앞으로는 위 용어들을 하나씩 정리해 볼 예정입니다.
```python

```
