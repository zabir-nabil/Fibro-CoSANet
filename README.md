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
2.  Download the kaggle.json from Kaggle account. [Kaggle authentication](https://www.kaggle.com/docs/api)
3.  Keep the kaggle.json file inside data_download folder.
3. `sudo cp kaggle.json /root/.kaggle/`
4. `python dataset_download.py` (make sure kaggle API is set up)
5. `sudo apt install unzip` if not installed already
6. `unzip osic-pulmonary-fibrosis-progression.zip` 

### Training

1. Set the training hyperparameters in `config.py`
2. Slope Prediction
   * To train **slopes model** run `python train_slopes.py`
   * trained model weights and results will be saved inside `hyp.results_dir`
3. Quantile Regression
   * To train **qreg model** run `python train_qreg.py`
   * trained model weights and results will be saved inside `hyp.results_dir`
