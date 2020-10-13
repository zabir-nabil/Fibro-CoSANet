# osic-pulmonary-fibrosis-progression

### Installation

1. Install Anaconda
2. `conda create -n pulmo python==3.7.5`
3. `conda activate pulmo`
4. `conda install -c intel mkl_fft`
5. `conda install -c intel mkl_random`
6. `conda install -c anaconda mkl-service`
7. `pip install -r requirements.txt`

### Download Dataset

1. `cd data_download`
2. `python dataset_download.py` (make sure kaggle API is set up)
3. `unzip osic-pulmonary-fibrosis-progression.zip`

### Training

1. Set the training hyperparameters in `config.py`
2. Slope Prediction
   * To train **slopes model** run `python train_slopes.py`
   * trained model weights and results will be saved inside `hyp.results_dir`
