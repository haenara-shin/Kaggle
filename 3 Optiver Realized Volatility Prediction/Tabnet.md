# TabNet Overview
- [TabNet Overview](#tabnet-overview)
  - [Introduction](#introduction)
  - [Overview](#overview)
    - [Main Concept](#main-concept)
    - [AutoEncoder](#autoencoder)
  - [Encoder](#encoder)
    - [Encoder Overview](#encoder-overview)
    - [Feature Transformer](#feature-transformer)
    - [Attentive Transformer](#attentive-transformer)
    - [Sparsemax](#sparsemax)
    - [Attentive Transformer Code Snippets](#attentive-transformer-code-snippets)
  - [Semi-supervised Learning](#semi-supervised-learning)
  - [Code](#code)
    - [Ref](#ref)
#
## Introduction 
- TabNet은 정형 데이터의 훈련에 맞게 설계된, `딥러닝 기반` 네트워크.
- 보통 정형 데이터의 훈련/예측 모델은 XGBoost/LightGBM/CatBoost 같은 Gradient Boosting Tree 알고리즘을 사용함. 딥러닝 모델은 성능이 떨어지거나 비용이 많이 들었음. 물론 예외는 있는데, 
  - (1) 100만 달러의 상금이 걸린 Netflix 영화 추천 문제에서 우승을 차지한 BellKor팀의 멤버인 Michael Jahrer의 Autoencoder 기반 딥러닝 아키텍처가 Porto Seguro 챌린지에서 [Private Leaderboard (LB) 1위](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629)를 차지함. 딥러닝 단일 모델은 아니지만, 딥러닝 기반 모델이 주요 컨셉이고 이를 위한 RankGauss normalization 기법 사용.
  - (2) XRD 데이터 바탕으로 결정 구조 예측하는 In-class Kaggle 챌린지에서 데이터 증강법을 사용하지 않고 순수한 딥러닝 모델을 사용해서 데이터 증강법 및 머신러닝 모델을 방법들을 모두 이기고 Private LB 1위한 내 모델!ㅋㅋ

#
## Overview
#
### Main Concept
> input feature에 대해 훈련 가능한 마스크(trainable mask)로 sparse feature selection을 수행
- 즉, TabNet의 feature selection은 특정 feature들만 선택하는 것이 아닌, 마치 linear regression 처럼 `각 feature에 가중치를 부여`하는 것. 설명 가능한(explainable) 모델이 되는 것임.
- ![tabnet](https://user-images.githubusercontent.com/58493928/131164437-84ddd1a1-9a08-429e-b08a-7dd47e0266a5.png)
  - trainable mask는 dense mask가 아니라 sparse mask임. 하지만 이는 input feature의 차원이 많을 수록 훈련 비용이 증가하고 훈련이 어렵다. (딥러닝 기반 네트워크의 공통된 문제..) 따라서 수천 개 이상의 feature들로 구성된 데이터에 대해서는 곧바로 TabNet 적용하면 별로.
#
### AutoEncoder
- TabNet의 구조는 Encoder-Decoder를 거쳐, 결측값들을 예측할 수 있는 AutoEncoder 구조임. 따라서 결측값들이 포함되어도 별도의 전처리 없이 값들을 채울 수 있음.

#
## Encoder
### Encoder Overview
- 인코더 구조
  - <img width="671" alt="assets-ml--MMXoGIZpcHkxIKjt_fv--MMXoN3Xl7JkuUo9xRLn-tabnet1" src="https://user-images.githubusercontent.com/58493928/131177231-3b45b468-5845-4b08-8b83-06e8609b625c.png">
- 인코더는 여러 decision step들로 구성됨.
  - decision steps 내의 2 가지 주요 블록들 
    > `Feature Transformer` & `Attentive Transformer`
  - 각 decision step에서 가장 중요한 `output은 (sparse) Mask`
1. `Feature transformer`블록에서 `임베딩(Embedding)`을 수행하고,
2. `Attentive Transformer`블록에서 `trainable mask`를 생성함.
    - 마스크의 활용도
      1. Feature importance 계산
      2. 이전 decision step의 feature에 곱해져서 `Masked feature` 생성
           - Masked feature는 다음 decision step의 input feature가 되며, 이전 decision step에서 사용된 mask의 정보를 피드백 하기때문에 feature의 재사용 빈도를 제어할 수 있음.
      3. 다음 decision step의 mask에서 적용될 `prior scale term` 계산
    - ![스크린샷 2021-08-27 오전 11 42 38](https://user-images.githubusercontent.com/58493928/131174343-631b4b7a-00f3-442f-a006-ea5058e04ad7.png)
      - Input feature의 dimension은 (# batch * # features). 매 decision step 마다 반복하기 전 Initial decision step(decision step 0) 에서는 별도의 mask feature 없이 BatchNormalization만 수행함.
        - 실제로 구현된 BN은 Ghost BN임.(전체가 아닌 샘플링 데이터에 대해서는 BN 수행)
      - Feature Transformer 블록에서 인코딩을 수행함.
        - Step >= 1 인 경우, 인코딩된 결과에서 ReLU layer를 거쳐서 해당 step의 decision output을 생성함. 향후, 각 step의 decision output 결과를 합산하여 overall decision 임베딩($d_{out}$)을 생성할 수 있고, 이 임베딩이 FC layer를 거치면 최종 output (classification/regression 예측 결과, $\hat y$)이 산출됨.
        - $d_{out} = \sum_{i=1}^{N_{steps}}$ReLU($d[i]$), $\hat y = W_{final}d_{out}$
        - 또한, ReLU layer의 결과에서 # of hidden unit 채널의 값들을 모두 합산하여 해당 decision step의 Feature importance의 결과를 합산하면, 최종 Feature importance(feature attributes)를 산출할 수 있음.
      - 인코딩된 결과는 Attentive Transformer 블록을 거쳐 mask를 생성함. Mask에 Feature를 곱하여 Feature selection이 수행되기 때문에 mask의 차원은 (# of batch * # of feture)임.
        - Attentive Transformer 블록 내에서 FC -> BN -> Sparsemax 를 순차적으로 수행하면서 Mask를 생성함.
      - Mask는 이전 decision step의 Feature와 곱하여 Masked feature를 생성함. Sparsemax 함수를 거쳤기 때문에 일반적인 tabular 데이터에서 수행하는 Hard feature selection이 아닌, `Soft feature selection`임. Masked feature는 다시 feature transformer로 연결되면서 decision step이 반복됨.
#
### Feature Transformer
#
### Attentive Transformer
#
### Sparsemax
#
### Attentive Transformer Code Snippets
- 구글 공식 코드
```python
### Attentive transformer
if ni < self.num_decision_steps - 1:

    # Determines the feature masks via linear and nonlinear
    # transformations, taking into account of aggregated feature use.
    mask_values = tf.layers.dense(
        features_for_coef,
        self.num_features,
        name="Transform_coef" + str(ni),
        use_bias=False)
    mask_values = tf.layers.batch_normalization(
        mask_values,
        training=is_training,
        momentum=self.batch_momentum,
        virtual_batch_size=v_b)
    mask_values *= complemantary_aggregated_mask_values
    mask_values = tf.contrib.sparsemax.sparsemax(mask_values)

    # Relaxation factor controls the amount of reuse of features between
    # different decision blocks and updated with the values of
    # coefficients.
    complemantary_aggregated_mask_values *= (
        self.relaxation_factor - mask_values)

    # Entropy is used to penalize the amount of sparsity in feature
    # selection.
    total_entropy += tf.reduce_mean(
        tf.reduce_sum(
            -mask_values * tf.log(mask_values + self.epsilon),
            axis=1)) / (
                self.num_decision_steps - 1)

    # Feature selection.
    masked_features = tf.multiply(mask_values, features)

  # Visualization of the feature selection mask at decision step ni
  tf.summary.image(
      "Mask_for_step" + str(ni),
      tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
      max_outputs=1,
  )
```

#
## Semi-supervised Learning

#
## Code
- [구글의 공식 코드](https://github.com/google-research/google-research/blob/master/tabnet/tabnet_model.py)와 이를 개선한 Modified TabNet, [PyTorch-TabNet](https://github.com/dreamquark-ai/tabnet)이 가장 유명.
    ```python
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

    clf = TabNetClassifier()  #TabNetRegressor()
    clf.fit(
    X_train, Y_train,
    eval_set=[(X_valid, y_valid)]
    )
    preds = clf.predict(X_test)
    ```

### [Ref](https://housekdk.gitbook.io/ml/ml/tabular/tabnet-overview)