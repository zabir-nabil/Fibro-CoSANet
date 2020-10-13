from kaggle.api.kaggle_api_extended import KaggleApi
from shutil import copyfile

# move kaggle.json file
# copyfile("kaggle.json", "/root/.kaggle")

api = KaggleApi()
api.authenticate()

print('authentication done')

# api.competition_download_files('osic-pulmonary-fibrosis-progression')

import os
os.system("kaggle competitions download -c osic-pulmonary-fibrosis-progression")

print('done')
