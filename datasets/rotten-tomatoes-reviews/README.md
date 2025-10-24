## Rotten tomatoes reviews, OMDB plots and posters dataset

# raw
Download dataset from [Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset/data) and put the csv files under [raw](raw).

# prep
Dataset in the prep folder is created by running notebook scripts [notebooks/data_prep/prep_rotten_tomatoes_data.ipynb](../../notebooks/data_prep/prep_rotten_tomatoes_data.ipynb), 
[notebooks/data_prep/create_omdb_plots_data.ipynb](../../notebooks/data_prep/create_omdb_plots_data.ipynb), 
[notebooks/data_prep/create_omdb_posters_data.ipynb](../../notebooks/data_prep/create_omdb_posters_data.ipynb)
in order. 
SQLite database is created by running [src/data/sqlite_database.py](../../src/data/sqlite_database.py).

The final processed data files are:
- `reviews_w_movies_full.csv`: Contains reviews along with movie metadata from both rotten tomatoes 
(origin Kaggle dataset) and Imdb (retrieved from OMDB). Each row is a review. There are reviews for 
8075 unique movies.
- `movie_plots.csv`: Contains movie plots (retrieved from OMDB) along with the same metadata columns as above. 
Each row is a movie plot. There are plots for 6257 unique movies, a subset of the movies with reviews.
- `movie_posters.csv`: Contains movie poster paths (retrieved from OMDB) along with the same metadata columns as above. 
Each row is a movie poster path. There are poster paths for 6079 unique movies, a subset of the movies with reviews.
- `poster` directory containing poster jpg files.
- `movies_meta.db`: SQLite database with indexes for movies metadata information.
