[Paper](https://arxiv.org/abs/1506.01497)
# Abstract
- 기존 object-detection 방법은 selective-search 알고리즘에 의존. 하지만 이 부분이 병목 구간이 됨. 
  - `RPN(Region Proposal Network)` 제안
    - 기존 방법론에서 사용하는 feature extraction 부분을 공유하여 사용하기 때문에 비용 증가 많지 않음.
    - 온전히 1개의 convolutional network 로 동시에 아래의 2가지를 출력함.
      - 1. 여러 바운딩 박스와
      - 2. 각 바운딩 박스에 오브젝트가 포함될 가능성(score)
    - end-to-end 로 학습이 가능하고, 기존 방법론인 Fast-RCNN 에 인풋으로 들어감.
#
## R-CNN (Region-based CNN)
- Selective search 로 바운딩 박스 후보를 여러개 만들어서 CNN에 넣어 classification
  1. selective-search를 이용해 입력된 이미지에서 약 2,000개 정도의 바운딩 박스 후보를 생성
  2. CNN에 넣고 돌릴 수 있도록 crop&resize
  3. 각 후보 영역을 CNN에 넣어서 feature vector를 추출
  4. feature vector를 각 클래스마다 학습된 binary SVM에 넣어서 classification
  5. 학습 시켜둔 바운딩 박스 regression 모델을 이용해 좌표 교정
- 병목 구간
  1. selective search
  2. 제안된 후보 영역에 대해서 feature extraction 을 수행해야함 ~2*k회(k: 클래스 수)
  3. 단계 마다 공유하는 computation 없음

#
## Fast-RCNN
- 전체 이미지의 Feature map 에서 region에 해당하는 위치의 feature 만 뽑아주는 RoI pooling 연산 제안. 모든 바운딩 박스 후보 마다 반복적으로 하던 feature extraction 연산을 한 번만 하게 됨.
  1. selective-search를 이용해 2,000개 정도의 후보 영역 생성
  2. pretrained CNN 모델의 끝단 maxpooling 대신 RoI-Pooling 레이어로 대체하여 후보영역들의 feature map 들이 나오게 함.
  3. 기존에 k개의 class를 classification 하던 마지막 FC layer를 FC + Softmax with k+1 레이어로 바꿔서 출력함 (+1은 배경을 의미하는 클래스)
  4. 2개의 브랜치로 나눠서
     1. 브랜치1: 각 RoI 마다 k+1개의 probability 출력
     2. 브랜치2: 각 k개의 클래스에 대해서 바운딩 박스 regression 수행, 바운딩 박스를 transform 할 (offset, scale)을 출력
- 병목 구간: selective-search

#
## Faster-RCNN
- selective-search를 RPN 을 만들어 대체
- RPN은 feature map을 입력으로 받아서 2종류의 출력 tensor를 만들어냄.
    1. cls layer를 통과시켜 얻은 2k scores 출력
        - 2k scores: 256-d가 cls layer(Conv, Kernel:1x1, filter:2K)를 통과한 출력값. 바운딩 박스에 물체가 있는지 없는지 2개 클래스의 확률 분포를 나타냄.
    2. reg layer를 통과시켜 얻은 4k coordinates 출력
        - 4k coordinates: 256-d가 reg layer(Conv, Kernel:1x1, filter:4k)를 통과한 출력값. 각 바운딩 박스를 어떻게 교정할지 나타내는 벡터로 구성된 텐서.
