# Vision Transformer (ViT)
[Ref](https://arxiv.org/pdf/2010.11929.pdf)
- Computer Vision 문제에 Transformer 구조를 사용한 CNN architecture 등장. 주로 자연어 처리에서 많이 사용됐고(평정했고) vision task에 적용했더니 역시 SOTA 모델 등극.
- ![스크린샷 2021-08-06 오전 11 09 09](https://user-images.githubusercontent.com/58493928/128553805-b08bb9c7-ce00-426f-84a1-647f9330a1ff.png)
- 기존의 제한적인 Attention mechanism에서 벗어나, CNN 구조 대부분을 Transformer로 대체함. (입력(input)단인 Sequence of Image Patch에서만 제외)
- 대용량 데이터셋을 pre-train --> Small image dataset(ImageNet-1k, Cifar100)에서 Transfer learning: 훨씬 적은 계산 리소스로 우수한 결과를 얻음. 단, 많은 데이터를 사전 학습해야됨.

## Summary
- 이전 vision task에서 self-attention 적용 한계
  - Self-attention을 적용하려는 시도는 있었지만, H/W accelerators에 비효율적 --> ResNet 구조가 SOTA가 됨. --> 기존의 Transformer를 최대한 그대로 적용하고자 함.
- 여기서 잠깐!! - Attention is All you Need - 
    - NLP에서 Self-attention을 활용한 Transformer. 
    - BERT는 Large dataset(Corpus)를 사전학습(pre-train) 시킨 후 작은 task 에서 fine-tuning 함.
- `Transformer` 장점: 계산 효율성(Efficiency) & 확장성(Scalability)
  - 100B parameter도 학습 가능!
  - 데이터셋이 크면 클수록 모델을 키워도 되며, 성능이 포화(Saturate)될 징후 안보임. 즉, 데이터셋이 크면 클 수록 성능도 더 높아짐!
- `Transformer 적용`은 어떻게?
  - 이미지를 PATCH로 분할 후 Sequence로 입력
  - NLP에서 단어(word)가 입력되는 방식과 동일(논문 제목이 "Image is worth 16x16 Words" 임을 잊지 말자)
  - Supervised learning 방식으로 학습
- `Transformer 특징`
  - ImageNet과 같은 Mid-size 데이터셋으로 학습시, ResNet 보다 낮은 성능을 보임(ImageNet이 더 이상 큰 데이터셋이 아님....그래서 본 Kaggle에서 Backbone 모델로 사용했을때 EfficientNet 보다 성능이 좋지 못했음.)
  - JFT-300M 사전 학습 후, transfer learning 하면 SOTA
  - Transformer 는 [inductive biases](https://robot-vision-develop-story.tistory.com/29)가 없음. 즉, Locality 와 [Translation equivariance](https://kmhana.tistory.com/27) 같은 CNN의 특성이 없음.
    - 1. CNN은 지역성(Locality)이라는 가정을 활용하여 공간적(Spatial) 문제를 품.
    - 2. RNN은 순차성(Sequentiality)이라는 가정을 활용하여 시계열(Time) 문제를 품.

## ViT 구조
![스크린샷 2021-08-06 오전 11 35 07](https://user-images.githubusercontent.com/58493928/128556567-2ce24786-981a-4b0a-8f7b-6d8bb52b6d28.png)
- 1. 입력 Embedding
  - Token embedding을 1D sequence로 입력
  - Patch Embedding이 출력됨
- 2. [CLS] Token
  - BERT의 [class] Token 처럼, 학습 가능한 Embedding patch 추가
- 3. Classification head
  - Pre-training: 1-hidden layer인 MLP
  - Fine-tuning: 1-linear layer
- 4. Position embedding
  - Patch embedding의 position 정보를 유지하기 위해서.
  - 2D-position embedding을 추가했는데 별로...이미지 이지만 1D Position embedding 사용(ㅠㅠ)
- 5. Transformer
  - Output embedding sequence는 encoder의 입력으로 들어감
  - Transformer encoder
    - Multi-head로 구성된 self-attention 메커니즘 적용
    - MLP block
    - 모든 block 이전에 LayerNormalization(LN) 적용
    - Residual Connection 모든 블록 끝에 적용.
- 6. Inductive bias
  - CNN(Locality 가정), RNN(Sequentiality 가정) 경우에 Global 한 영역의 처리는 어려움.
  - ViT는 일반적인 CNN과 다르게 공간에 대한 'inductive bias'가 없음.
  - 따라서, ViT는 더 많은 데이터를 통해, 원초적인 관계를 Robust 하게 학습시켜야 함.
    - ViT는 MLP layer에서만 Local 및 Translation Equivariance 함.
    - Self-Attention mechanism is Global!
  - 2차원 구조 매우 드물게 사용
    - 이미지 patch를 잘라서 넣는 input 부분
    - Fine-tune 시, 다른 해상도(resolution)의 이미지를 위한 위치 임베딩 조정 --> 공간(Spatial) 관계를 처음부터 학습해야함 (왜냐하면 POSition embedding 초기화 시, 위치 정보 전달 X)
- 7. Hybrid Architecture
  - Image patch의 대안으로, CNN의 feature map의 Sequence를 사용할 수 있음.
    - CNN의 feature 는 Spatial size가 1x1이 될 수 있음.
    - CNN의 feature 추출 --> Flatten --> Embedding projection 적용

## Implementation
- [Google-Research](https://github.com/google-research/vision_transformer): Google-Research 에서 keras로 손쉽게 사용할 수 있도록 해놓음. Pretrained models 제공 (MLP mixer도 있다!)
- [keras_implementation](https://github.com/faustomorales/vit-keras)
  - 주의: output shape이 conv2d와 다름. GAP을 붙일 수 없다. Flatten 하거나 아님 바로 FC 붙임. (내가 잘못 알고 있을수도?)
  
    ```python
    !pip install vit-keras

    from vit_keras import vit, utils

    image_size = 384
    classes = utils.get_imagenet_classes()
    model = vit.vit_b16(
        image_size=image_size,
        activation='sigmoid',
        pretrained=True,
        include_top=True,
        pretrained_top=True
    )
    url = 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Granny_smith_and_cross_section.jpg'
    image = utils.read(url, image_size)
    X = vit.preprocess_inputs(image).reshape(1, image_size, image_size, 3)
    y = model.predict(X)
    print(classes[y[0].argmax()]) # Granny smith
    ```
    - Visual attention mapping
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from vit_keras import vit, utils, visualize

    # Load a model
    image_size = 384
    classes = utils.get_imagenet_classes()
    model = vit.vit_b16(
        image_size=image_size,
        activation='sigmoid',
        pretrained=True,
        include_top=True,
        pretrained_top=True
    )
    classes = utils.get_imagenet_classes()

    # Get an image and compute the attention map
    url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
    image = utils.read(url, image_size)
    attention_map = visualize.attention_map(model=model, image=image)
    print('Prediction:', classes[
        model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
    )  # Prediction: Eskimo dog, husky

    # Plot results
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(image)
    _ = ax2.imshow(attention_map)
    ```

### 총평
- 사전학습 데이터에 성능이 좌우 되기 때문에 아직은 EfficientNetB#이 가장 현실적으로 보임.