## Rotten tomatoes reviews dataset

# raw
Download dataset from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset/data) and put the csv filed under raw.

# prep
Dataset in the prep folder is created by scripts [notebooks/data_prep/prep_rotten_tomatoes_data.ipynb](../../notebooks/data_prep/prep_rotten_tomatoes_data.ipynb), 
[notebooks/data_prep/create_omdb_plots_data.ipynb](../../notebooks/data_prep/create_omdb_plots_data.ipynb) 
in order.

It cleans the raw data, and joins the reviews with the movie meta-data, creates plots data from Omdb, and adds all movie metdata 
to both files. Final data is one file with reviews and one file with movie plots.
