# Parallel_Normalization_joblib
* Using python library-joblib to do parallel computing on data normalization.  
* This a final project of the course **NTU Parallel and Distributed Programming 2016 Fall**.
* For more detail of this project concept, please refer to `PDP_fn.pdf`.

## File Description
* `pdpaux.py`: contain parallel and non-parallel data normalization function.
  - With any size of matrix, if each row is variable and column is data, then you can use this function to normalize data.
* `main.py`: my testing code on my parallel function performance.
* `data.csv`: the data I used.
* `PDP_fn.pdf`: report of this project.

## Executing 
1. Your machine should install joblib. Open terminal and type  
`sudo pip install joblib`
1. After successful install, put the three file:`pdpaux.py`, `main.py`, `data.csv` in the same directory, and type  
`python main.py`
1. Then, you can see result like this  
```
    data dimension: (17, 92160)
    Parallel execution time:        5.584
    Non-Parallel execution time:    9.983
    result checking:                pass
```

## Executing Your Own Data
If you want to use your own data to test. You can modify the read data part of `main.py`.  
Use my data normalization API to test performance.  
* `pdpaux.normalizeDataParallel(X,2)`  
* `pdpaux.normalizeData(X)`  
