from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

print('authentication done')

# api.competition_download_files('osic-pulmonary-fibrosis-progression')

import os
os.system("kaggle datasets download khoongweihao/osic-model-weights")

print('done')
