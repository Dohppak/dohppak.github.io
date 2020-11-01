---
title: "Use Pretrained Model with torchvision"
date: "2020-10-31"
path: /blog/pretrained
tags: Vision, Image Classifciation Model Serving
layout: post
---

# DeepLearning For everyone!

Deep learning is rapidly becoming a key technology in the application field of artificial intelligence (LeCun et al. 2015). In the meantime, deep learning has shown tremendous achievements in various fields such as computer vision, natural language processing, and speech recognition. As a result, interest in deep learning continues to grow. However, some hurdles are challenging to implement these deep learning papers directly or understand their implementation. First of all, it requires enormous data and storage space and various computational resources (GPU, TPU). For example, Bert uses a huge Corpus of 330 million words **(800 million words of BookCorpus data and 2.5 billion words of Wikipedia data)**. Also, the GPU used for training is huge. In the base model case, 8 chapters of V100 must be trained for 12 days, and in the case of the large model, 64TPU chips are trained for 4 days. This is equivalent to the amount of learning 280 V100s per day. There is a very high cost to build such an infrastructure.

Also, at this point, most applications do not have as many computational resources as expected. However, the actual application model is not the data or the GPU, but **the weight of the model** that is actually and used.

----

딥러닝은 인공지능의 응용 분야에서 핵심 기술로 빠르게 자리를 잡아가고 있습니다 (LeCun et al. 2015). 그동안 딥러닝은 컴퓨터 비전, 자연어 처리, 그리고 음성인식 등의 여러 분야에서 실로 대단한 성과들을 보여주었고, 이에 따라 딥러닝에 대한 관심은 계속해서 높아지고 있습니다. 하지만 이러한 딥러닝 논문들을 직접 구현하거나, 구현체를 이해하는데 매우 어려운 허들이 존재하게 됩니다. 우선 막대한 데이터와 저장공간 그리고 다양한 컴퓨테이셔널 리소스(GPU, TPU)가 필요하게 됩니다. 예를들어 Bert 같은 경우는 총 3.3억 단어 **(8억 단어의 BookCorpus 데이터와 25억 단어의 Wikipedia 데이터)** 의 거대한 Corpus를 사용합니다. 또한 학습에 사용되는 GPU 또한 거대합니다. Base 모델의 경우 V100 8장을 12일 동안 학습해야하며, Large 모델의 경우는 64TPU Chips를 4일동안 학습합니다. 이는 V100의 280개를 하루동안 학습하는 양과 동일합니다. 이러한 인프라 스트럭쳐를 구축하기 위해서는 매우 큰 비용이 발생하게 됩니다. 

또한 이 때 대부분의 응용 분야는 생각보다 많은 컴퓨테이셔널 리소스를 가지지 못한 경우가 많습니다. 하지만 실제 응용 분야에서 사용하게 되는 모델은 Data나 GPU가 중요한 것이 아니라, 실제로 학습이 완료되고 사용하게 되는 모델의 Weight입니다.

# What is Model Weight?


To understand the weight of a model, we first need to know the environment in which the training takes place. If training is a process of creating a deep learning model, after the training is completed, we apply an instance to the model and perform an inference process. In the learning phase, you need to build the big data mentioned above, knowledge of the model architecture, and the Computational Infrastructure used for learning. The model architecture represents the data as a high-dimensional latent. If this latent representation is suitable for the learning task, it will perform better. At this time, while repeating Forward and Back-Propagation, we learn the model's weight by the loss created by the data. These weights can be seen as converting the data we have into knowledge and converting it to the model. Then, if we stop learning and have the weight of the given model, we can get answers to the knowledge of the task we learned in the learning phase (e.g., is this sentence positive or negative, what document is classified?). For data in the same format!

모델의 Weight를 이해하기 위해서는 우선 학습이 이루어지는 환경을 알아야합니다. Training이 Deep Learning Model을 Creating 하는 과정이라면, 이후 학습이 완료된 이후에는 우리는 모델에 Instance를 적용하여 Inference를 하는 과정을 하게 됩니다. 학습 단계에서는 위에서 언급한 큰 데이터와 모델 아키텍쳐에 대한 지식, 그리고 학습에 사용될 Computational Infrastructure를 구축해야합니다. 데이터들은 모델 아키텍쳐에 의해 고차원 latent로 표현되며, 이 latent representation이 학습 테스크에 적합하다면 더욱 좋은 성능을 내게 됩니다. 이떄 Forward와 Back-Propagation을 반복하면서 우리는 데이터가 만들어 내는 Loss에 의해서 모델의 Weight를 학습하게 됩니다. 이 Weight들은 사실상 우리가 가지고 있는 데이터를 지식으로 변환하여 모델에게 전환했다고 볼수 있습니다. 그렇다면 학습을 멈추고 우리가 주어진 모델의 Weight를 가지고 있다면, 학습단계에서 학습했던 Task에 대한 지식들 (예를들면 이 문장이 긍정적인가 부정적인가, 어떤 문서로 분류되는가?)에 대한 답면을 얻을수 있습니다. 동일한 형식에 데이터에 대해서는 말이죠!

<img src="../img/train_inference.png" width = 80%>

# Inference 프로세스

Deep learning inference is the process of predicting previously unseen data using a trained DNN model. DNN models created for image classification, natural language processing, and other AI tasks can be large and complex due to tens or hundreds of layers of artificial neurons and the millions or billions of weights that connect them. The larger the DNN, the more compute, memory, and energy it takes to run it. The longer the response time (or "wait time") from entering data into the DNN to receiving the result.

Therefore, in deep learning research, much research is being done on the efficiency of the model and the performance of the model.

---

딥 러닝 추론은 훈련 된 DNN 모델을 사용하여 이전에 보지 못한 데이터를 예측하는 프로세스입니다. 이미지 분류, 자연어 처리 및 기타 AI 작업을 위해 생성 된 DNN 모델은 수십 또는 수백 층의 인공 뉴런과 이들을 연결하는 수백만 또는 수십억 개의 가중치로 인해 크고 복잡 할 수 있습니다. DNN이 클수록이를 실행하는 데 더 많은 컴퓨팅, 메모리 및 에너지가 소비되며 DNN에 데이터를 입력 할 때부터 결과를받을 때까지 응답 시간 (또는 "대기 시간")이 길어집니다.

때문에 딥러닝 연구에서는 모델의 퍼포먼스 만큼이나 모델의 효율성에 대해서도 많은 연구가 이루어지고 있습니다.

# Tutorial!

In example, we will try how to get the desired result by using the pre-trained model for the task in Image Classification.

We will focus on using a pre-trained model to predict the label of the input, so we will also discuss the process involved in this. This process is called model inference. The whole process consists of the following main steps:

### Process

- Perform transformations on the image (e.g., resize, crop center, normalize, etc.)
- Forward Pass: Find the output vector using pre-trained weights. Each element of this output vector describes the confidence that the model predicts an input image that belongs to a particular class.
- Display predictions based on the scores obtained.

-----

오늘 예시에서는 Image Classification에 Task에 대해서 우리가 사전 학습된 모델을 활용하여 원하는 결과를 얻는 방법을 해보고자 합니다. 

입력의 레이블를 예측하기 위해 사전 학습 된 모델을 사용하는 방법에 중점을 둘 것이므로 이와 관련된 프로세스에 대해서도 논의하겠습니다. 이 프로세스를 모델 추론이라고합니다. 전체 프로세스는 다음과 같은 주요 단계로 구성됩니다.

입력 이미지 읽기
- 이미지에 변형 수행 (예 : 크기 조정, 가운데 자르기, 정규화 등)
- Forward Pass : 사전 훈련 된 가중치를 사용하여 출력 벡터를 찾습니다. 이 출력 벡터의 각 요소는 모델이 특정 클래스에 속하는 입력 이미지를 예측하는 신뢰도를 설명합니다.
- Confindence Score를 기반으로 예측을 표시합니다.

### 1.0 Download Torch Model

Torchvision supports various pre-trained models. These pre-trained models are neural network models trained on large benchmark data sets such as ImageNet. The deep learning community has benefited a lot from these open-source models, which is also one of the main reasons for the rapid development of computer vision research. Other researchers and practitioners can use these state-of-the-art models instead of inventing everything from scratch.


```python
import torch
from torchvision import models
```


```python
dir(models)
```
    ['alexnet',
     'densenet',
     'densenet121',
     'densenet161',
     'densenet169',
     'densenet201',
     'detection',
     'googlenet',
     'inception',
     'inception_v3',
     'mnasnet',
     'mnasnet0_5',
     'mnasnet0_75',
     'mnasnet1_0',
     'mnasnet1_3',
     'mobilenet',
     'mobilenet_v2',
     'resnet',
     'resnet101',
     'resnet152',
     'resnet18',
     'resnet34',
     'resnet50',
     'resnext101_32x8d',
     'resnext50_32x4d',
     'segmentation',
     'shufflenet_v2_x0_5',
     'shufflenet_v2_x1_0',
     'shufflenet_v2_x1_5',
     'shufflenet_v2_x2_0',
     'shufflenetv2',
     'squeezenet',
     'squeezenet1_0',
     'squeezenet1_1',
     'utils',
     'vgg',
     'vgg11',
     'vgg11_bn',
     'vgg13',
     'vgg13_bn',
     'vgg16',
     'vgg16_bn',
     'vgg19',
     'vgg19_bn',
     'video',
     'wide_resnet101_2',
     'wide_resnet50_2']



`dir(models)` 명령어를 확인해 보면 우리가 현재 사용할수 있는 pretrained 모델의 종류를 알 수 있습니다. 본 포스팅에서는 저희는 대표적인 모델 resnet50을 사용합니다. resnet은 input으로 들어오는 X를 Convolution Block에 의해 학습된 Feature와 다시 결합함으로서 다양한 관점에서 데이터를 바라보는 시각을 제시합니다. 관련된 코드는 아래 첨부되어 있습니다.

----

Use the `dir(models)` command to see what kind of pre-trained models we currently have available. In this post, we use the representative model resnet50. Resnet presents the perspective of viewing data from various perspectives by combining X coming through the input with the feature learned by the Convolution Block again. The relevant code is attached below.


```python
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):  #@save
    """The Residual block of ResNet."""
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
```


```python
resnet50 = models.resnet50(pretrained=True)
```
```
    Downloading: "https://download.pytorch.org/models/resnet50-19c8e357.pth" to /Users/seungheondoh/.cache/torch/checkpoints/resnet50-19c8e357.pth
    100%|██████████| 97.8M/97.8M [00:10<00:00, 9.38MB/s]
```

우리는 `pretrained=True` 라는 명령어를 통해 https://download.pytorch.org/models/resnet50-19c8e357.pth 에서 사전에 학습되어 있는 weight를 다운로드 받을 수 있습니다. 이때 거대한 데이터와 Computational Infra가 100M가 안되는 지식으로 압축되어 있는 것을 확인 할 수 있습니다.

----


Once we have the model with us, the next step is to transform the input image with specific format. If the data domain of the input is different, the result cannot be obtained significantly. (When sentence input is inserted in the vision model or voice input is inserted)


```python
from torchvision import transforms

transform = transforms.Compose([            
    transforms.Resize(224),                 
    transforms.ToTensor(),                  
    transforms.Normalize(                   
        mean=[0.5, 0.5, 0.5],
        std=[0.229, 0.224, 0.225]               
    )
])
```

We are going to use a simple image called blue_brid for the image. Can our model really recognize this blue bird?


```python
from PIL import Image
img = Image.open("blue_bird.jpg")
img
```
![png](output_16_0.png)


```python
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
```

The model is converted to a specific format by the function created above. We change the image above to the input shape used in the model, 224, and then normalize it for better performance. After that, it is made in batch units through the unsqueeze function. For now, batch is 1. 

Then we have to declare that the function no longer needs to learn through a function called torch.eval. If we put the image in the model, we can get the output corresponding to 1xClass Number.


```python
resnet50.eval()
with torch.no_grad():
    out = resnet50(batch_t)
    
out.shape
```

```
    torch.Size([1, 1000])
```


# Image Net Classes 

The Image Net dataset is a matter of classifying 1000 images. Meta annotations corresponding to 1000 images can be downloaded from the site below. Then, let's find out which image our model classified the instance image into through the largest value among the values corresponding to each index of each output.

you can download in https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json


```python
import json
class_idx = json.load(open("imagenet_class_index.json",'r'))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
```


```python
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
```


```python
_, indices = torch.sort(out, descending=True)
infernece = [(idx2label[idx], percentage[idx].item()) for idx in indices[0][:5]]
```


```python
infernece
```

```
    [('indigo_bunting', 99.34516906738281),
     ('bee_eater', 0.2820887863636017),
     ('jacamar', 0.03670770302414894),
     ('macaw', 0.02816816233098507),
     ('European_gallinule', 0.0279614869505167)]
```


99% are classified as indigo_bunting. It's not bad performance!

reference
- https://github.com/huggingface/transformers
- https://www.learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
- https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/


```python

```
