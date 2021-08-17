# EfficientNet
- [2019](https://arxiv.org/pdf/1905.11946.pdf)
- Summary & Conclusion: 파라미터 수는 줄이고, 성능은 압도적으로 올림!
  - 모델의 성능 = resolution * depth * width --> 세 구성 요소의 최적 비율을 찾음.
  - ![스크린샷 2021-08-05 오후 9 42 22](https://user-images.githubusercontent.com/58493928/128457098-30f644a7-ea8f-45a1-aa7a-d47416a2b64b.png)
- 배경: ResNet, InceptionNet 계열이 주류가 된 이후, 데이터와 parameters 를 많이 넣은 모델이 ImageNet LB를 압도하고 있을때, 
  - (1) 적은 파라미터로 효율적인 성능을 내려는 InceptionNet의 초기 이념, 
  - (2) 쉬운 architecture 구성으로 높은 성능을 내고자 했던 ResNet의 초기 이념이 결합됨. 
  - (3) NasNet 처럼, Neural Architecture Search - AutoML을 사용해서 기본 architecture 골격을 만들었음. 

## Introduction
- 효율적이며 확장가능한 높은 성능의 CNN 모델
1. 네트워크의 `Depth`, `Width`, `Resolution` 간의 balance를 통해 효과적이며 좋은 성능을 얻음.
   1. `Depth`, `Width`, `Resolution` 로 구성된 복합 계수 제안(Uniformly scales all dimensions of depth/width/resolution)
   2. ResNet과 MobileNet에도 적용 가능함. (확장 가능성이 높음)
   3. Transfer Learning에서도 더 적은 수의 parameters로 SOTA 달성 --> Detection 분야에서 `EfficientDet` 등장

## CNN modeling
1. CNN Acc.
   1. parameter가 커지면서 더 높은 정확도를 얻고 있으나, 
   2. 그에 따라 특수한 병렬 처리(different accelerator) 필요성 대두 --> 자원의 한계 상황에 도달함.
2. CNN Eff.
   1. CNN은 종종 over-parameterized 
   2. 모델 사이즈를 줄이거나 Neural Architecture Search를 사용
      1. Hand-craft 모델보다 종종 더 효율적이며, Fine-tuning 불필요
      2. 작은 모델에는 NasNet 방법이 적합할 수 있으나, 큰 모델에 적용할 수 있을지는 불분명 --> 한계.
3. Model scaling
   1. Depth(# of layers), Width(# of channels), Resolution(Image size) 고려
   2. Image size 증가는 더 높은 FLOPs
   3. Depth와 Width는 CNN의 표현력과 관련됐지만, 어떻게 효율적으로 scaling 할 수 있을지 밝혀지지 않음. --> 실험!

## Scaling up CNN
- 기존의 성능 향상 방안: 대표적인 방법은 모델의 '크기'를 키우는 것. 하지만 '어떻게' 모델의 크기를 키울 것인가? 에 대한 연구 부족.
  - 그래서 기본적으로
  - (1) Depth: # of layers
  - (2) Width: # of filters
  - (3) Resolution: Image size
  - 를 수동으로(사람이 노가다로..) scale 조절해서 sub-optimal 의 성능을 얻음.
- 그렇다면, `model scale-up`의 원리는 무엇인가?
  - `depth`, `width`, `resolution` 간의 균형(balance)이 중요함!!
  - 단순한 비율(고정된 계수)을 통해 적합한 균형을 얻을 수 있음.
    - Assumption: Computational resolution이 2**N 만큼 증가할 경우, 적정 scaling up 비율?
    - depth, width, image size를 각각 $\alpha$<sup>N</sup>, $\beta$<sup>N</sup>, $\gamma$<sup>N</sup> 만큼 증가시킴!! ($\alpha$, $\beta$, $\gamma$는 사전에 정해진 각각의 고정 계수)
    - $\alpha$ = 1.2, $\beta$ = 1.1, $\gamma$ = 1.15로 했을때 FLOPS이 2배에 가까우면서 최고의 성능이 나왔고 이를 baseline 모델(B0)로 정함. 이후부터 FLOPS가 2<sup>2</sup> 가 되는 모델을 B1, 2<sup>3</sup> 이 되는 모델을 B2...B7까지 만들었음. [참고](https://tw0226.tistory.com/29)
- Scaling-up 효과에 대한 직관적인 관점
  - `해상도가 커지면`, Receptive field를 키우기 위해서 `더 많은 layer가 필요`함.
  - `해상도가 커지면`, 더 세밀한 패턴을 capture 하기 위해서 `channel을 키워야` 함.
  - ![스크린샷 2021-08-05 오후 10 06 34](https://user-images.githubusercontent.com/58493928/128458788-fa098072-7851-409d-b6b1-ba98aa4b638d.png)
  - ResNet 및 MobileNet에서도 Compound scaling-up 개념이 잘 작동하지만, model scaling의 효율성은 어떤 architecture를 사용하느냐에 영향을 받음.
    - 저자는 `NasNet(Neural Architecture Search)`를 사용해서 'New Baseline Network'을 개발함. 이를 바탕으로 Scaling-up 하면서 family network를 만듦.

## Compound Model Scaling
- Scaling을 formulation 해서 문제를 해결 하고자 함.[수식 쓰는게 귀찮아서 링크를 연결함](https://kmhana.tistory.com/26)
![스크린샷 2021-08-05 오후 11 02 07](https://user-images.githubusercontent.com/58493928/128463621-49a9164f-75d0-468b-82fa-d2dc654bd3a1.png)
![스크린샷 2021-08-05 오후 11 03 30](https://user-images.githubusercontent.com/58493928/128463704-ef775070-84c0-4f6e-937f-474c200e72c6.png)
- Scaling dimensions: 각 Dimension(width,depth,resolution)은 서로에게 영향을 주며, 서로 다른 리소스 제약 내에서 영향을 받음. 각 dimension이 커질수록 정확도가 향상되지만, 성능 향상 폭은 점차 감소되거나 심지어 낮아질 수도 있음.
  - `Width`
    * 특징 1. 작은 모델의 사이즈를 키울 때 많이 사용됨.
    * 특징 2. 세밀한(fine-grained) feature를 capture 하기 위해 사용됨.
    * 단점 1. Shallow 모델(layer가 작은 모델)에서, 상위 레벨의 복합적인 feature를 파악하기 어려움.
    * 단점 2. Width가 넓어짐에 따라 빠르게 saturation 됨.
  - `Depth`
    * 장점: Capture Richer and More complex feature
    * 단점: Skip-connection 이나 BatchNorm 으로 완화하고 있지만, vanishing gradient로 학습하기 어려움. 특정 층 이상 쌓이면, 성능이 저하됨.
  - `Resolution`
    * 장점 1. 세밀한(fine-grained) feature를 capture 하기 위해 많이 사용됨
    * 장점 2. Detection 처럼 복잡한 task 에서 더 높은 resolution 사용
    * Resolution의 증가는 정확도에 영향을 줌.
      * 왜?? 입력의 해상도가 작을수록 CNN을 거치면서 - Convolution, Pooling - 특징이 소실되거나, 추출되지 않는 경우가 발생하기 때문.
    * 단점: Resolution 증가할수록, 성능 증가폭은 감소됨. (이게 단점인가?)
 - ![스크린샷 2021-08-05 오후 11 29 31](https://user-images.githubusercontent.com/58493928/128466258-949a349e-f910-41a8-be04-2d443da4bd54.png)
   - (left) Width(channel) 증가; (mid) depth(layer) 증가; (right) resolution(image size) 증가
   - log 함수 처럼 우측으로 갈 수록 어느정도 수렴함.
## Compound scaling 필요성
- 실험적으로, 더 큰 Resolution은 더 큰 Receptive field 가 필요하다 (layer 증가 필요). 왜냐하면 resolution이 커질수록 유사한 픽셀의 영역이 증가하기 때문!
- 더 큰 resolution은 fine-grain pattern capture가 필요하다 (channel 증가 필요). 왜냐하면 Fine-grain pattern과 Channel의 크기(width) 연관
- 단일 차원의 scaling이 아닌, 복합적인 차원의 균형 및 조정이 필요하다는 직관&결론..
- ![스크린샷 2021-08-05 오후 11 14 27](https://user-images.githubusercontent.com/58493928/128464772-dd4ce1c9-eba4-4300-baf1-6fb2e5d6c871.png)
- ![스크린샷 2021-08-05 오후 11 16 27](https://user-images.githubusercontent.com/58493928/128464975-077e9492-27ed-43b5-b887-f07311a16458.png)
- 저자는 [Multi-object Neural Architecture Search](https://kmhana.tistory.com/26)를 통해 New Mobile-size Baseline(`EfficientNet-B0`) 제안
  - 기본 block: MBConv + Squeeze-and-Excitation optimization
  -  SiLU(Swish-1) & Stochastic Depth(짧은 network로 학습하고 - 깊이가 단축됨. 마치 Drop-out처럼. 각 Layer의 하위 그룹(Subset)이 무작위로 제외되고, skip-connection을 통해 loss가 우회됨. - 추론에서는 깊은 network 사용) & AutoAugment 사용(pass..)
