![SIIM-COVID19 Github Banner](https://user-images.githubusercontent.com/58493928/128546381-97883f11-a61a-411e-8497-3471d0aed890.png)

# [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/overview)
- 2021.May.17 ~ 2021.August.09
- Host
  - FISABIO: The foundation for the promotion of health and biomedical research of Valencia Region
  - RSNA: Radiological Society of North America 
- Evaluation
  - The challenge uses the standard PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.5. 
  - In this competition, we are making predictions at both a study (multi-image) and image level.
    - Study-level labels: Studies in the test set may contain more than one label. They are as follows:
        ```
        "negative", "typical", "indeterminate", "atypical"
        ```
    - For each study in the test set, you should predict at least one of the above labels. The format for a given label's prediction would be a class ID from the above list, a confidence score, and 0 0 1 1 is a one-pixel bounding box.
    - Image-level labels: Images in the test set may contain more than one object. For each object in a given test image, you must predict a class ID of "opacity", a `confidence` score, and bounding box in format `xmin` `ymin` `xmax` `ymax`. If you predict that there are NO objects in a given image, you should predict `none 1.0 0 0 1 1`, where `none` is the class ID for "No finding", 1.0 is the confidence, and `0 0 1 1` is a one-pixel bounding box.

  - Submission File: The submission file should contain a header and have the following format:
    ```
    Id,PredictionString
    2b95d54e4be65_study,negative 1 0 0 1 1
    2b95d54e4be66_study,typical 1 0 0 1 1
    2b95d54e4be67_study,indeterminate 1 0 0 1 1 atypical 1 0 0 1 1
    2b95d54e4be68_image,none 1 0 0 1 1
    2b95d54e4be69_image,opacity 0.5 100 100 200 200 opacity 0.7 10 10 20 20
    etc.
    ```
- Prizes: Awarded on the basis of private leaderboard rank. Only selected submissions will be ranked on the private leaderboard.
    ```
    1st Place - $30,000
    2nd Place - $20,000
    3rd Place - $10,000
    4th Place - $8,000
    5th Place - $7,000
    6th - 10th Places - $5,000 each
    ```
- Code Requirements
  - Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:
    ```
    CPU Notebook <= 9 hours run-time
    GPU Notebook <= 9 hours run-time
    Internet access disabled
    Freely & publicly available external data is allowed, including pre-trained models
    Submission file must be named submission.csv
    ```
- Data
  - In this competition, we are identifying and localizing COVID-19 abnormalities on chest radiographs. This is an object detection and classification problem.

  - For each test image, you will be predicting a bounding box and class for all findings. If you predict that there are no findings, you should create a prediction of "none 1 0 0 1 1" ("none" is the class ID for no finding, and this provides a one-pixel bounding box with a confidence of 1.0).

  - Further, for each test study, you should make a determination within the following labels:
    ```
    'Negative for Pneumonia' 'Typical Appearance' 'Indeterminate Appearance' 'Atypical Appearance'
    ```
  - To make a prediction of one of the above labels, create a prediction string similar to the "none" class above: e.g. `atypical 1 0 0 1 1`
  - The images are in DICOM format, which means they contain additional data that might be useful for visualizing and classifying.
- Dataset information
  - The train dataset comprises 6,334 chest scans in DICOM format, which were de-identified to protect patient privacy. All images were labeled by a panel of experienced radiologists for the presence of opacities as well as overall appearance.
  - Note that all images are stored in paths with the form `study/series/image`. The `study` ID here relates directly to the study-level predictions, and the `image` ID is the ID used for image-level predictions. The hidden test dataset is of roughly the same scale as the training dataset.
- Files
    - `train_study_level.csv` - the train study-level metadata, with one row for each study, including correct labels.
    - `train_image_level.csv` - the train image-level metadata, with one row for each image, including both correct labels and any bounding boxes in a dictionary format. Some images in both test and train have multiple bounding boxes.
    - `sample_submission.csv` - a sample submission file containing all image- and study-level IDs.
  - Columns 
    - in `train_study_level.csv`
      - `id` - unique study identifier
      - `Negative for Pneumonia` - `1` if the study is negative for pneumonia, `0` otherwise
      - `Typical Appearance` - `1` if the study has this appearance, `0` otherwise
      - `Indeterminate Appearance`  - `1` if the study has this appearance, `0` otherwise
      - `Atypical Appearance`  - `1` if the study has this appearance, `0` otherwise
    - in `train_image_level.csv`
      - `id` - unique image identifier
      - `boxes` - bounding boxes in easily-readable dictionary format
      - `label` - the correct prediction label for the provided bounding boxes
  - `kaggle competitions download -c siim-covid19-detection`.
