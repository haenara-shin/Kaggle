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

#
# Tips for Google Colab 
- Open the developer tools in chrome or firefox, and then paste the below codes.
```javascript (Prevent from disconnection in Google Colab)
function ClickConnect(){
console.log("Working"); 
document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect,60000)
```

# 
# [14 Tips to save RAM memory for 1+GB dataset](https://www.kaggle.com/pavansanagapati/14-simple-tips-to-save-ram-memory-for-1-gb-dataset)(kor)
- 주피터 노트북으로 `pandas`를 사용해서 small dataset(100MB)을 다룰때는 성능 저하가 거의 없지만, 1GB 이상의 큰 데이터를 다룰때는 메모리 부족으로 실행이 불가능할 때가 있음.
	- `Spark`는 100GB or TB의 큰 데이터를 다룰 수 있지만 비싸고(연구용으로 적합하지 않음), `pandas`와 같은 기능이 부족함.
- 파이썬의 효율적인 RAM memeory usage 사용법에 대해서 알아보자. (`pandas`와 관련)

	1. Free Memory using `gc.collect()`
	- 파이썬의 garbage collection module 사용
	```python
	import gc
	del something_your_references/values
	gc.collect()

	```

	2. Datatype conversion
		```python
		df.info(memory_usage='deep') 

		```
	- 각 데이터 타입은 `pandas.core.internals.module`의 특별한 클래스를 가짐. `pandas`는 `ObjectBlock class`를 사용해서 `string columns` 포함하는 블럭을 나타내고, `FloatBlock class`를 사용해서 `float columns`를 포함하는 블럭을 나타냄. 숫자값(numeric values - integers and floats)을 나타내는 블럭들에 대해서, `pandas`는 columns를 결합해서 `Numpy ndarray` 형태로 저장함. `Numpy ndarray`는 C 배열 기반이므로 `continuous block of memory`에 저장이 되고, 따라서 값들에 대한 slice 접근이 굉장히 빠르게 일어나며 더 적은 공간을 사용함. 이처럼 각 데이터 타입은 개별적으로 저장되기 때문에 데이터 타입별 메모리 사용 상황을 체크해보자.
		```python
		for dtype in ['float','int','object']:
		    selected_dtype = df.select_dtypes(include=[dtype])
		    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
		    mean_usage_mb = mean_usage_b / 1024 ** 2
		    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))
		```
	- 숫자형 표현 (Numeric values)
		- 데이터의 여러 subtype ('`unit8`', '`int8`', '`int16`'...) 존재함. 예를 들어, '`int8`'는 값을 저장할때 1byte(8bits)를 사용하고 256 값들을 binary로 나타낼 수 있다. 하지만 positive values만 저장하기 때문에 `uint (unsigned integers)` 가 `int (signed integers)` 보다 더 효율적임. 
		- 당연히 더 작은 bit 숫자로 표현하는게 더 공간 절약(가능하다면).
		```python
		int_types = ["uint8", "int8", "int16"]
		for it in int_types:
	    	print(np.iinfo(it))

	    # We will be calculating memory usage a lot,so will create a function save some resource.
		def mem_usage(pandas_obj):
		    if isinstance(pandas_obj,pd.DataFrame):
		        usage_b = pandas_obj.memory_usage(deep=True).sum()
		    else: # we assume if not a df it's a series
		        usage_b = pandas_obj.memory_usage(deep=True)
		    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
		    return "{:03.2f} MB".format(usage_mb)
		
		# int - unsigned comparison
		gl_int = df.select_dtypes(include=['int'])
		converted_int = gl_int.apply(pd.to_numeric,downcast='unsigned')
		print(mem_usage(gl_int))
		print(mem_usage(converted_int))
		
		compare_ints = pd.concat([gl_int.dtypes,converted_int.dtypes],axis=1)
		compare_ints.columns = ['Before','After']
		compare_ints.apply(pd.Series.value_counts)

		# float 64 - 32
		gl_float = df.select_dtypes(include=['float'])
		converted_float = gl_float.apply(pd.to_numeric,downcast='float')
		print(mem_usage(gl_float))
		print(mem_usage(converted_float))
		compare_floats = pd.concat([gl_float.dtypes,converted_float.dtypes],axis=1)
		compare_floats.columns = ['Before','After']
		compare_floats.apply(pd.Series.value_counts)

		# total
		optimized_df = df.copy()
		optimized_df[converted_int.columns] = converted_int
		optimized_df[converted_float.columns] = converted_float
		print(mem_usage(df))
		print(mem_usage(optimized_df))
		```
	- Object type(string, etc) to `categoricals`
		- strings 는 fragmented way 로 저장됨. 즉, object column에 있는 각 element는 메모리 내에 실제 값의 위치에 대한 '주소'를 담고 있는 포인터임. 따라서 많은 메모리를 차지하게 됨. 각 포인터가 메모리의 1byte 를 차지하는 반면, 각 실제 string value는 같은 양의 메모리를 사용함. 
		![스크린샷 2021-08-05 오후 2 30 45](https://user-images.githubusercontent.com/58493928/128424012-e6708fd5-2608-4855-bd45-f5bcf752c494.png)
		- `pandas` version 0.15 부터 `categoricals` 제공함. 쉽게 말해서, integer values 갖고 있는 Raw values에 대해서 공간 절약 할 수 있음. 
			```The category type uses integer values under the hood to represent the values in a column, rather than the raw values. Pandas uses a separate mapping dictionary that maps the integer values to the raw ones. This arrangement is useful whenever a column contains a limited set of values. When we convert a column to the category dtype, pandas uses the most space efficient int subtype that can represent all of the unique values in a column.```
			```python
			df_rating = df.rating
			df_rating_cat = df_rating.astype('category')
			```
		- (*/주의/*) 공간 절약은 할 수 있지만, numerical value calculation은 할 수 없음.



	3. Import selective # of rows in data
	- 별건 아니고, `df = pd.read_csv('path', nrows=2000)` 처럼 nrows에 보고 싶은 숫자만 일단 넣어서 봄.

	4. Random row selection (random sampling of data)
	- 위에서 nrows 대신 skiprows 옵션을 사용함.
	```python
	def csv_file_length(fname):
	    process = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	    result, error = process.communicate()
	    if process.returncode != 0:
	        raise IOError(error)
	    return int(result.strip().split()[0])

	random_rows_selection = csv_file_length('path')
	print('Number of random rows in "path" is:', random_rows_selection)

	skip_rows = np.random.choice(np.arange(1, random_rows_selection), size=random_rows_selection-1-10000, replace=False)

	skip_rows=np.sort(skip_rows)
	print('Rows to skip:', len(skip_rows))
	print('Remaining rows in the random sample:', random_rows_selection-len(skip_rows))

	train = pd.read_csv('path', skiprows=skip_rows)
	train.head()

	del skip_rows
	gc.collect()
	```

	5. Skip # of rows
	- 옵션에 header 없이 skiprows를 걸어준다. 근데 이러면 header 정보가 다 없어지는데? 이미 알고 있는 경우에만 사용.
	```python
	df = pd.read_csv("path",skiprows=2000 ,header=None,nrows=20000)
	```
	또는
	```python
	df = pd.read_csv("path",skiprows=range(1,2000),nrows=20000)
	```

	6. Use of Generators (`yield` 와 `next`사용)
	- Generators allow you to create a function that returns one item at a time rather than all the items at once. This means that if you have a large dataset, you don’t have to wait for the entire dataset to be accessible.

	- Generator functions allow you to declare a function that behaves like an iterator. It allows us to make an iterator in a fast, easy, and clean way. An iterator is an object that can be iterated (looped) upon. It is used to abstract a container of data to make it behave like an iterable object. A generator looks a lot like a function, but uses the keyword yield instead of return. 

	- In summary when we call a generator function or use a generator expression, we return a special iterator called a generator. We can assign this generator to a variable in order to use it. When we call special methods on the generator, such as next(), the code within the function is executed up to yield.

	- The advantage lies in the fact that generators don’t store all results in memory, rather they generate them on the fly, hence the memory is only used when we ask for the result. Also generators abstract away much of the boilerplate code needed when writing iterators, hence also helps in reducing the size of code.

	7. Eliminate unnecessary loops by using itertools (`itertools` 사용하자. 근데 왜 코테에서는 금지 시켜..)

	8. Do not use `+` operator for strings (문자열 연결할 때 + 연산자 쓰지 말것!!)
	- String은 immutable. 따라서 string에 뭔가 더할때마다 파이썬은 새로운 string을 만들기 때문(== 새로운 주소 == 새로운 메모리 할당 필요).
	```python
	mymsg = "Micheal"
	msg="My name is %s . I live in US"% mymsg
	# msg="My name is "+mymsg+". I live in US"
	```

	9. Memory profiling
	- 줄이는 테크닉은 아니고 모니터닝.
	- `%memit` 을 function 위에 넣는다. (`%timeit` 도 비슷하게 사용)
	```python
	%memit pi_calculation()
	def pi_calculation(n=50000000) -> "area":
	    """Estimate pi with monte carlo simulation.
	    
	    Arguments:
	        n: Number of Simulations
	    """
	    return np.sum(np.random.random(n)**2 + np.random.random(n)**2 <= 1) / n * 4
	```

	10. Memory leaks (함수형 프로그래밍)
	- temporary variables를 삭제함. (`del something_your_variables`) 하지만 이렇게 하지 말고, 함수를 만들어서 함수 내에서 과정을 처리하게끔 하면 intermediate varaibles 은 함수 종료와 동시에 자동 삭제 됨.

	11. Line profiler (`%lprun`)
	- 어떤 부분이 hot spot 인지 알게 됨. (high cost of CPU-time 등을 알게 해서 최적화가 필요한 부분에 대해 생각하게함.)
	```python
	!pip install line_profiler
	%load_ext line_profiler
	...your function...
	%lprun -f your_function your_function()
	```
	- 색깔을 입히려면
	```python
	!pip install py-heat-magic
	%load_ext heat
	%%heat #(at the top of the cell)
	```

	12. Memory profiler
	- check the memory usage of the interpreter at every line. The increment column allows us to spot those places in the code where large amounts of memory are allocated. 
	- Array 작업 많이 할때 유용. 불필요한 array 생성과 복사는 프로그램 성능(속도) 저하 유발하기 때문.
	```python
	!pip install memory_profiler
	%load_ext memory_profiler
	%%writefile memscript.py
	def your_function():
		return something

	from memscript import your_function
	%mprun -T mprof0 -f your_function your_function()

	print(open('mprof0', 'r').read())
	```

	13. `autoreload` module
	- The module reloads the code before each execution. Once you get locked into a TDD loop and you start refactoring code from the notebook into additional files, this module will reload the code in the additional files. (Kaggle 처럼 다른 input 파일/디렉토리 가져다 쓸때 유용)
	```python
	%load_ext autoreload
	%autoreload 2
	```

	14. Increase default memory limit size 
	- Jupyter notebook has a default mem. limit size. You can try to increase the memory limit by following the stpes
	- (1) Generate Config file using command
	```python
	jupyter notebook --generate-config
	
	```
	- (2) Open jupyter_notebook_config.py file situated inside 'jupyter' folder and edit the following property. Remember to remove the '#' before the property value.
	```python
	NotebookApp.max_buffer_size = your_desired_value
	
	```
	- (3) Save and run the jupyternotebook. It should now utilize the set memory value. Also, don't forget to run the notebook from inside the jupyter folder! 

	- Or, alternatively, you can simply run the Notebook using below command
	```python
	jupyter notebook --NotebookApp.max_buffer_size=your_desired_value
	
	```

#
# List
1. [Predicting crystal structure from X-ray diffraction](https://www.kaggle.com/c/nano281fa2020/overview) - Completed_1/34 (Inclass competition)
2. [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/c/siim-covid19-detection/overview) - Completed_97/1324 (Top 8% - Bronze medal)