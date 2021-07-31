# Kaggle

```bash
|-- 1_Structured_Predicting crystal structure from X-ray diffraction(completed_1/34)
|-- 2_ObjectDection_SIIM-FISABIO-RSNA COVID-19 Detection(progressing_218/1165)

```

```
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/haenara-shin/DACON.git
git push -u origin main
```

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

# Tips for Google Colab 
- Open the developer tools in chrome or firefox, and then paste the below codes.
```javascript (Prevent from disconnection in Google Colab)
function ClickConnect(){
console.log("Working"); 
document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect,60000)
```


# List
1. [Predicting crystal structure from X-ray diffraction](https://www.kaggle.com/c/nano281fa2020/overview) - Completed_1/34
2. [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/overview) - Progress
