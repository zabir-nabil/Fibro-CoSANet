from kaggle.api.kaggle_api_extended import KaggleApi
import os

api = KaggleApi()
api.authenticate()

print('authentication done')

# api.competition_download_files('osic-pulmonary-fibrosis-progression') # this may fail


os.system("kaggle competitions download -c osic-pulmonary-fibrosis-progression")

print('done')
