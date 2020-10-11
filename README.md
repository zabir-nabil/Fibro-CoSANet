# osic-pulmonary-fibrosis-progression

### Installation

1. Install Anaconda
2. `conda create -n pulmo python==3.7.5`
3. `conda activate pulmo`
2. `pip install -r requirements.txt`

### Download Dataset

1. `cd data_download`
2. `python dataset_download.py` (make sure kaggle API is set up)

### Training

1. Set the training hyperparameters in `config.py`
2. To train **slopes model** run `python train_slopes.py`
