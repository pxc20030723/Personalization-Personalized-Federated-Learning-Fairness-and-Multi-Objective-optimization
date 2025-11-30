from itertools import count
import os
import argparse
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np


MOVIELENS_100k_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def download_movielens(data_dir: str = "./data"):
    """
    Download and extract MovieLens-1M dataset.
    
    Args:
        data_dir: Directory to save the dataset
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    zip_path = data_path / "ml-100k.zip"
    extract_path = data_path / "ml-100k"
    
    # Check if already downloaded
    if extract_path.exists():
        print(f"Dataset already exists at {extract_path}")
        return str(extract_path)
    
    # Download
    print(f"Downloading MovieLens-100k from {MOVIELENS_100k_URL}...")
    urllib.request.urlretrieve(MOVIELENS_100k_URL, zip_path)
    print(f"✓ Downloaded to {zip_path}")
    
    # Extract
    print(f"Extracting to {data_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_path)
    print(f"✓ Extracted to {extract_path}")
    
    # Clean up zip file
    zip_path.unlink()
    print(f"✓ Cleaned up zip file")
    
    
    return str(extract_path)
'''
def compute_entropy(dic):
    probs=[p for p in dic.values() if p >0]
    entropy=-sum(p*np.log(p) for p in probs)
    return entropy
'''
def analyse_users_preference(min_rating_time,rating_dir='./data/ml-100k/u.data',item_dir='./data/ml-100k/u.item'):
    '''
    step1: merge ratings and movie_info into one file like
              user_id  movie_id  ratings  unknown  Action  ...  Romance  Sci-Fi  Thriller  War  Western
          0      196       242        3        0       0  ...        0       0         0    0        0
          1      186       302        3        0       0  ...        0       0         1    0        0

    step2: 
           input ratings with genres 
           output users' preferrence like
           user_id | primary_genre | primary_score | Action | Drama | Comedy | ...
                1  |    Action     |     0.65      |  0.65  |  0.20 |  0.15  | ...

    '''
    ratings = pd.read_csv(rating_dir,sep='\t',names=['user_id', 'movie_id', 'ratings'], usecols=[0,1,2])
    print(f"len(ratings):{len(ratings)}, users:{ratings['user_id'].nunique()}")
    genre_cols=['unknown', 'Action', 'Adventure', 'Animation', "Children's",
        'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western']
    movie_info=pd.read_csv(item_dir,sep='|', encoding='latin-1', names=['movie_id']+genre_cols, usecols=[0]+list(range(5,24)) )
    print(f"first row in movie_info:{movie_info.iloc[0]}")
    print(f"len(movie_info):{len(movie_info)}")
    #merge ratings and movie_info
    ratings_with_genres=ratings.merge(movie_info, on='movie_id', how='left' )
    print(f"shape of merged ratings:{ratings_with_genres.shape}")
    print(f"first 5 rows of merged ratings:\n{ratings_with_genres.head()}")

    #step2 start
    all_users=ratings_with_genres['user_id'].unique()
    user_preferences=[]
    for user_id in all_users:
        user_data=ratings_with_genres[ratings_with_genres['user_id']==user_id]
        user_data_filtered=user_data[user_data['ratings']>=3]

        user_avg_ratings=user_data_filtered['ratings'].mean()
        genre_count={}
        for genre in genre_cols:
            count=(user_data[genre]).sum()
            genre_count[genre]=count
 
        #normalize to percentage
        total_count=sum(genre_count.values())
        genre_percentages={g: count/total_count for g, count in genre_count.items()}
        primary_genre=max(genre_percentages, key=genre_percentages.get)
        primary_score=genre_percentages[primary_genre]
        user_pre={
                'user_id':user_id,
                'primary_genre':primary_genre,
                'primary_score':primary_score,
                'total_count_catagories':total_count,
                'avg_ratings':user_avg_ratings

            }
        user_pre.update(genre_percentages)
        user_preferences.append(user_pre)
    user_preferences_df=pd.DataFrame(user_preferences)
    numeric_clos=['primary_score', 'avg_ratings']+genre_cols
    user_preferences_df[numeric_clos]=user_preferences_df[numeric_clos].round(2)
    user_preferences_df.to_csv('./data/user_preferences_count.csv', index=False)


        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MovieLens-100k dataset')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to save dataset')
    parser.add_argument('--rating_dir', type=str, default='./data/ml-100k/u.data',
                        help='Directory of u_data')
    parser.add_argument('--item_dir', type=str, default='./data/ml-100k/u.item',
                        help='Directory of u_item')
    parser.add_argument('--min_rating_time', type=int, default=5,
                        help='user whose rating times less than 5 would be discarded')
    args = parser.parse_args()
    download_movielens(args.data_dir)
    analyse_users_preference(args.min_rating_time,args.rating_dir,args.item_dir)