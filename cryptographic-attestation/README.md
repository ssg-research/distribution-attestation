
# Boneage and Census in CrypTen

## Setup
The training was tested with Python 3.7.16. \
Install requirements from requirements.txt: \
```pip install -r requirements.txt```

### Installing CrypTen
CrypTen needs to be installed manually from this [CrypTen fork](https://github.com/Tobias512/CrypTen) because it fixes some errors and adds fetures that are necessary for the training.
1. Clone repository and check out branch "main+dev":
```
git clone https://github.com/Tobias512/CrypTen.git
git fetch origin
git checkout dev+main
```
2. Build and install CrypTen:
```
pip install -r requirements.txt
python setup.py build
python setup.py install
```
3. Copy config to CrypTen Package:
```
cp configs/default.yaml [path to python interpreter]/site-packages/crypten-0.4.0-py3.7.egg/configs/
```
For example, when using Anaconda with an enviorment named "CrypTen" the path looks as follows:
```
cp configs/default.yaml ~/anaconda3/envs/CrypTen/lib/python3.7/site-packages/crypten-0.4.0-py3.7.egg/configs/
```
4. Verify installation:
```
python crypten_test.py
```

## Running the Training

The code is set up to either run on a single or multiple GPUs on a single machine. \
The training is carried out by three parties each running as a separate process. Two parties are the computation parties that do the training and the third party is a helper. \
All aspects of training are controlled by command line arguments and most are set to reasonable defaults. Most arguments don't need to be changed.

### Single GPU
Running the training on a single GPU is done by running the following command. For Boneage:
```
python bone_model_training.py --gpu=0 --crypten --multiprocess
```
For Census:
```
python census_training.py --gpu=0 --crypten --multiprocess
```
By changing the gpu argument the training can be run on a different GPU.

### Multiple GPUs
Running the training on multiple GPU one a single machine is done by running the following command. For Boneage:
```
python distributed_launcher.py --world_size=2 --gpus=0,0,0 bone_model_training.py --distributed --crypten
```
For Census:
```
python distributed_launcher.py --world_size=2 --gpus=0,0,0 census_r_training.py --distributed --crypten
```
The gpu argument is a comma seperated list and by specifies which GPUs to run the processes on. The world_size specifies the number of computation parties.
The number of GPUs must be world_size + 1 to account for the TTP.
