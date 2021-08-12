# What to do (COVID19 detection)
 * Identify and localize COVID-19 abnormalities on chest radiographs. In particular, you'll categorize the radiographs as negative for pneumonia or typical, indeterminate, or atypical for COVID-19.
 * Classfication model + Detection model

# How to do
 - At the study level: 
 	- Use classification model to train, you can try these models: `EfficientNetbX(X=0~7)`, `EfficientNetV2`, `ViT`, `ResNet101V2`
 	- I selected the `efficientnetv2-l-21k-ft1k` of `EfficientNetV2` in the TF hub.
 		- The model architectures with no suffixes are pretrained on ImageNet1K. The ones with the '21k' as the suffix are pretrained on ImageNet21K and the ones with '21k-ft1k' as the suffix are pretrained on ImageNet21K and then finetuned on ImageNet1K.
 		- top1=86.9%, which is the highest score among them (`efficientnetv2-xl-21k-ft1k` looks having higher top1 score, but the model is heavier and the actual tested score was not enough to beat `efficientnetv2-l-21k-ft1k`)

 - At the image level:
 	- To predict 2-class image level, I selected `EfficientNetB7`(`EfficientNetV1`) with various training conditions and then ensembled them. This step is to classifiy the `opacity` and `none`.
 	- To make the ensemble, I added the weight for each pre-trained model, but it didn't affect much. 
	- To detect the image level, use detection model to train, you can try these models: `RetinaNet`, `Yolov3`, `Yolov4`, `Yolov5`, `Faster-RCNN`, `Cascade-RCNN`, `EfficientDet`
	- I selected the `Deconformal Convolutional Networks` which is based on the `Faster-RCNN`. `Cascade-RCNN` was also comparable to `DCN(Faster-RCNN)`, but `DCN(Faster-RCNN)` is slightly lighter and higher mAP than `Cascade-RCNN`.

 - Finally merge the forecast results and generate submission.csv

# 총평
 - 처음 참여해본 캐글 오픈 대회 (GAN 프로젝트 팀원의 추천)
 - 이미지 분류/탐지 대회. 최근 computer vision 공부 중이라 mmdetection 실습해볼 목적으로 참가했음.
 - 실제로 열심히 참여한 시간은 약 4주 남짓인데, 처음이라 캐글 노트북/커널을 어떻게 사용하는지, 버전은 어떻게 관리하는지 등 시행착오가 많았음. 그래서 처음에 자원(TPU/GPU resource) 낭비가 컸음 ㅠㅠ. 
 - 대회 종료할때 쯤 되서 어떻게 접근해야 더 성능을 개선할 수 있을지 아이디어들이 떠올랐음.
 - 결정적으로 `discussion` 탭을 적극적으로 살펴보면서 다른 사람들의 아이디어나 노하우를 적극 채용했어야 함. 
 	* 특히, loss function 으로 `aux loss`를 사용했어야 했는데 아무 생각 없이 cross-entropy만 주구장창 사용했음 ㅠㅠ
 	* [aux loss](https://www.kaggle.com/c/siim-covid19-detection/discussion/263676) 에 대한 좋은 설명. UCSD 수학과 연구 교수 인듯...천재인가...아무튼, 쉽게 말해서 multi-task learning을 도입(여기서는 classification 에 segmentation head를 더함). [Github 라이브러리](https://github.com/qubvel/segmentation_models)가져다 쓰면 손쉽게 쓸 수 있다.
 	```Python
 	import segmentation_models as sm

	build_model():
	    base_model = sm.FPN(BACKBONE, encoder_weights='imagenet', 
	       input_shape=(None, None, 3), classes=3, activation='sigmoid')

	    x = base_model.get_layer(name='top_activation').output 
	    x = tf.keras.layers.GlobalAveragePooling2D()(x)
	    x = tf.keras.layers.Dense(4, activation='sigmoid', name='out2')(x)

	    model = tf.keras.Model(inputs=base_model.input, 
	       outputs=[base_model.output,x]) 

	    opt = tf.keras.optimizers.Adam()
	    met1 = sm.metrics.iou_score
	    met2 = tf.keras.metrics.AUC(curve='PR',multi_label=True)
	    loss1 = sm.losses.bce_jaccard_loss
	    loss2 = tf.keras.losses.BinaryCrossentropy()

	    model.compile(loss={'sigmoid':loss1,'out2':loss2}, 
	              loss_weights = [1.0,1.0], optimizer = opt,
	              metrics={'sigmoid':met1, 'out2':met2}) 

	    return model
	```
 - 일반적인 데이터, 모델로는 (상위권 레벨에 도달하기 까지) 한계가 있음. Discussion tab을 잘 살펴봐야 함.
 	* `Image size` 을 늘려야함.
 	* `Noisy student training` (Generate OOF prediction soft-labels and use the soft-labels to train the model in the next cycle.) -- 중요한 팁.
 		* [Self-training with Noisy Student improves ImageNet classification_GoogleBrain+CMU](https://arxiv.org/pdf/1911.04252.pdf)
 			* 기존 EfficientNet 아키텍쳐에 ImageNet 데이터셋과 함께 '라벨을 붙이지 않은 대량의 이미지'를 self-training 기법을 적용하여 학습.
 			![스크린샷 2021-08-11 오후 8 48 57](https://user-images.githubusercontent.com/58493928/129135585-e4b51f4f-ffcb-4f8c-a8f5-a2c491631519.png)
	 			1. EfficientNet 아키텍쳐를 사용한 teacher network를 생성한 뒤, 이미지넷 데이터로 학습시킴
	 			2. 학습된 teacher network로 라벨이 매겨져 있지 않은 3억장의 이미지(JFT라는 데이터셋인데 구글 내부 데이터셋)에 라벨을 매김(pseudo-labeling on unlabeled data)
	 			3. teacher network 보다 크기가 같거나 큰 student network를 생성한 뒤, 이미지넷 데이터와 라벨을 매긴 데이터를 합쳐서 학습 시킴
	 			4. 학습된 student 모델을 teacher network로 삼고, 2-4 과정을 반복함.
 				* 3단계에서 student network를 학습시킬 때, RandAugment, Dropout, Stochastic Depth 세 종류의 Noise를 적용함. 이를 통해 좀 더 robust한 모델을 학습시킬 수 있음.
 					* `RandAugmentation`
 						- 인풋 이미지에 RandAugmentation을 적용함.
 						- 기존에 다양한 Data Augmentation을 모두 정리해 놓은 기법임. 회전/contrast/.. 14가지 augmentation 기법 모아놓고 '랜덤하게 N개를 뽑음'. 그리고 얼마나 강하게 변화를 줄 것인지를 나타내는 magnitude M을 각각의 기법들에 전달하여 augmented images를 얻음. 
 						```Python
 						transforms = [
 						'Identity', 'AutoContrast', 'Equalize', 'Rotate', 'Solarize', 'Color', 'Posterize', 'Contrast', 'Brightness', 'Sharpness', 'ShearX', 'ShearY', 'TranslateX', 'TanslateY']
 						def randaugment(N, M):
 							"""Generate a set of distrotions.

 							Args:
 								N: Number of augmentation transformations to apply sequentially.
 								M: Magnitude for all the transformations.
							"""
							sampled_ops = np.random.choice(transforms, N)
							return [(op, M) for op in sampled_ops]
						```
 						- 연구 결과 M은 10 ~ 15 사이가 최적 
 					* Deep Networks with Stochastic Depth
 						- RandAugment가 Noisy student 모델에 들어오는 인풋 값에 
 						Noise를 주었다면, 모델이 학습하는 과정에서는 `Dropout` 과 `Deep Network with Stochastic Depth` 방식으로 Noise를 줌.
 						- 즉, dropout(일부 노드들을 학습 중 꺼버리는것)을 레이어에 적용함. 각각의 레이어를 일정한 확률에 따라서 학습 과정에서 생략하고 넘어감. (수식 생략)
 					* 그 외 Data balancing, filtering
 						- unlabeled 데이터 전처리 기법.
 						- Unlabeled 데이터셋에서 클래스 별로 이미지 장 수의 균형을 맞추는 것. 부족한 클래스는 이미지를 복사해서 늘려주고, 너무 많은 클래스는 confidence(teacher network를 통과시켜 얻은 라벨의 스코어 값. 이 값이 낮을 수록 ImageNet 데이터셋의 클래스에 속하지 않는 out-of-domain 데이터일 가능성이 큼)가 높은 이미지들만 추려냄. 
 						- Filtering이란 이러한 confidence가 낮은 이미지를 걸러주는 것을 의미함. 따라서 Data balancing + Filtering을 통해 unlabeled 데이터셋의 이미지 분포가 최대한 labeled 데이터셋과 일치하도록 만들어줌!!
 					* [Ref](https://yeomko.tistory.com/42)
 - Part 1: Study-level
 	1. I used an ensemble of only 7 models from the same base architecture(`EfficientnetV2-l`). 
 		* `EfficientnetV2-m`, `EfficientnetB5, 7` 등 다른 모델들과 `ensemble` 했어야...역시 캐글은 앙상블이 전부인듯.
 	3. Models were trained on only 256 image sizes and different augmentations
 		* 이미지 크기가 클 수록 좋은데...[512, 720] 범위에서 학습 시키고 augmentation 도 엄청 적극적으로 했어야 함.
 	4. Segmentation as [aux loss](https://www.kaggle.com/c/siim-covid19-detection/discussion/263676) 사용했어야 함.
 	5. 각 모델 fine-tuning
 	6. Hflip Test Time Augmentation(Hflip TTA) 사용했어야 했음.

 - Part 2: 2Class(Binary) Study-level(`none` class prediction)
 	1. Used an emsemble of 10 models from the same base architecture(`EfficientNetB7`) with a different training condition and top layers. 하지만 여기서도 `EfficientNetV2-M` 과 `EfficientNetB6` 등을 앙상블 했어야 함. 기존에 내가 알고 있던 k-fold 숫자를 증가시키는건 거의 효과 없음. 5-fold가 최고였음.
 	2. 역시 aux loss 사용 안함...ㅠㅠ
 	3. Part1과 비슷한 약점. 동메달권 언저리에서 머무를 수 밖에 없었던 결정적인 이유임.

 - Part 3: Image-level (Detection - `opacity` predictions)
 	1. MMDetection 실습/구현을 위해서 참가했으니 절반의 성공임. 하지만 여기서도 모델 앙상블을 사용하는게 일반적임.
 	2. 다른 상위권 랭커들은 `EfficientDetD3`, `EfficientDetD5`, `Yyolov5x`, `Yolov5l6`, `Retinanet_x101_64x4d_fpn` 등을 5-fold model ensemble 함. 어마어마한 차이. mmdetection 으로는 한계였음. 혹시 mmdetection도 앙상블 했으면 달라졌을까?
 	3. 역시 이미지 사이즈 키웠어야 함. 랭커들 중 일부는 `896, 512, 620, 620, (1330, 800)` 사용
 	4. 또한, 상위권 랭커들은 다음과 같은 작업을 함. self-training 기법을 여기서도 사용 했어야 했을까?
 	```
 	We first trained d5, d3, yolov5x, yolov5l6 on only opacity predictions. We used WBF (iou=0.62) to generate pseudo lables of public test. We then used these labels to train d3, yolov5x, yolov5l6 models again.
	In the pseudo label training d3 was trained with ema and yolov5 models were trained with a few none images as well.
	```

[상위권 랭커_5th](https://www.kaggle.com/c/siim-covid19-detection/discussion/263945)
[상위권 랭커_11th](https://www.kaggle.com/c/siim-covid19-detection/discussion/263701)
