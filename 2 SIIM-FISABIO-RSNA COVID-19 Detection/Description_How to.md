# What to do
 * COVID19 detection
 * Classfication model + Detection model

# How to do
 - At the study level: 
 	- Use classification model to train, you can try these models: `EfficientNetbX(X=0~7)`, `EfficientNetV2`, `ViT`, `ResNet101V2`
 	- I selected the `efficientnetv2-l-21k-ft1k` of `EfficientNetV2` in the TF hub.
 		- The model architectures with no suffixes are pretrained on ImageNet1K. The ones with the '21k' as the suffix are pretrained on ImageNet21K and the ones with '21k-ft1k' as the suffix are pretrained on ImageNet21K and then finetuned on ImageNet1K.
 		- top1=86.9%, which is the highest score among them (`efficientnetv2-xl-21k-ft1k` looks having higher top1 score, but the model is heavier and the actual tested score was not enough to beat `efficientnetv2-l-21k-ft1k`)

 - At the image level:
 	- To predict 2-class image level, I selected `EfficientNetB7`(`EfficientNetV1`) with various training conditions and then ensembled them.
	- To detect the image level, use detection model to train, you can try these models: `RetinaNet`, `Yolov3`, `Yolov4`, `Yolov5`, `Faster-RCNN`, `Cascade-RCNN`, `EfficientDet`
	- I selected the `Deconformal Convolutional Networks` which is based on the `Faster-RCNN`

 - Finally merge the forecast results and generate submission.csv

 (작업중_2021.08.10)