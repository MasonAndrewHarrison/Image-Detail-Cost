import kaggle



kaggle.api.authenticate()
kaggle.api.dataset_download_files("programmer3/artistic-painting-dataset", path='dataset/', unzip=True)




