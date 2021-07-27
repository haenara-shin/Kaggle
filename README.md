# Kaggle
- Repos. for Kaggle I have participated in. (not in-class Kaggle competition)
- When you want to change and import the dataset from Kaggle notebooks to Colab notebooks, follow the below instructions.
	(1) In Colab, 
	```python
	!pip install fsspec
	!pip install gcsfs
	```
	(2) Restart the kernel of Colab
	(3) In Kaggle notebook
	```python
	import pandas as pd
	from kaggle_datasets import KaggleDatasets
	GCS_PATH = KaggleDatasets().get_gcs_path('siim-covid19-detection') #whatever you want to copy and paste input-dataset
	print(GCS_PATH) # to copy and paste
	```
	(4) Back to the colab notebook

	```python
	GCS_PATH = `copied and pasted address`
	df = pd.read_csv(GCS_PATH+'/train.csv')
	```

# List
(1) [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/overview) - Progress