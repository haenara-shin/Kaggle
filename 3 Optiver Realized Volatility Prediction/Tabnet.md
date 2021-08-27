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
- 구글의 공식 코드와 이를 개선한 Modified TabNet, [PyTorch-TabNet](https://github.com/dreamquark-ai/tabnet)이 가장 유명.
    ```python
    from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

    clf = TabNetClassifier()  #TabNetRegressor()
    clf.fit(
    X_train, Y_train,
    eval_set=[(X_valid, y_valid)]
    )
    preds = clf.predict(X_test)
    ```

[Ref](https://housekdk.gitbook.io/ml/ml/tabular/tabnet-overview)