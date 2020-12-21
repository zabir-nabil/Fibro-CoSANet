# osic-pulmonary-fibrosis-progression

### Installation

1. `git clone https://github.com/zabir-nabil/osic-pulmonary-fibrosis-progression.git`
2. `cd osic-pulmonary-fibrosis-progression`
3. Install Anaconda [Anaconda](https://www.anaconda.com/products/individual)
4. `conda create -n pulmo python==3.7.5`
5. `conda activate pulmo`
6. `conda install -c intel mkl_fft` (opt.)
7. `conda install -c intel mkl_random` (opt.)
8. `conda install -c anaconda mkl-service` (opt.)
9. `pip install -r requirements.txt`

### Download Dataset

1.  Download the kaggle.json from Kaggle account. [Kaggle authentication](https://www.kaggle.com/docs/api)
2.  Keep the kaggle.json file inside data_download folder.
3. `sudo mkdir /root/.kaggle`
4. `sudo cp kaggle.json /root/.kaggle/`
5. `sudo apt install unzip` if not installed already

 * `cd data_download; python dataset_download.py; mv osic-pulmonary-fibrosis-progression.zip ../../; unzip ../../osic-pulmonary-fibrosis-progression.zip; cd ../; python train_slopes.py`

### Training

1. Set the training hyperparameters in `config.py`
2. Slope Prediction
   * To train **slopes model** run `python train_slopes.py`
   * trained model weights and results will be saved inside `hyp.results_dir`
3. Quantile Regression
   * To train **qreg model** run `python train_qreg.py`
   * trained model weights and results will be saved inside `hyp.results_dir`

### Results

1. Slope model [all results](https://drive.google.com/drive/folders/1nV8Cc4u-YFSDxTtjJoUXxsh-k7A21Z73?usp=sharing)


