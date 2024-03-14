# SMNN
Scalable Monotonic Neural Networks


## Installation
0. Install `python` >= 3.9.13, Install `pip` >= 22.0.4


## activate virture environment 
1. make new virture environment and activate 
```make venv
pip -m venv [venv_name]
```
```move to venv directory
cd [venv_name]
```
```activate
.\Scripts\activate
```


## Requirements : numpy>=1.21.0, pandas>=1.4.2, scikit-learn>=1.0.2, scipy>=1.8.0, torch>=1.10.2, matplotlib>=3.5.2, 
2. To install requirements: 
```setup
pip install -r requirements.txt
```


## Training
3. To train the SMNN for each real-world dataset, run this command at each dataset-folder:
(The network size[structure] can be modified in "[dataset_name]_train.py" )

```move to "dataset_name" forder, ex)autompg
cd autompg
```
```train
python autompg_train.py --bs 128 --lr 0.005 --epochs 1000 
```


## Evaluation
4. To evaluate SMNN on (pre)trained model, run:
```eval
python autompg_test.py 
```
The output after running `python [dataset_name]_test.py ` is:

	Testing Data Size : 836
	total param amount: 27473
	Test Loss: 0.00600, Test mse: 4.70051


## Pre-trained Models
pretrained model is saved in each folder and if you want to evaluate other trained model, enable the code that allows you to save the model in "[dataset_name]_train.py"

