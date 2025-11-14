import os
import argparse
import urllib.request
import zipfile
from pathlib import Path
import pandas as pd


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

def analyse_users_preference(rating_dir='./data/ml-100k/u.data',item_dir='./data/ml-100k/u.item'):
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
    ratings_with_genres=ratings.merge(movie_info, left_on='movie_id', right_on='movie_id', how='left' )








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MovieLens-100k dataset')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory to save dataset')
    parser.add_argument('--rating_dir', type=str, default='./data/ml-100k/u.data',
                        help='Directory of u_data')
    parser.add_argument('--item_dir', type=str, default='./data/ml-100k/u.item',
                        help='Directory of u_item')                  

    args = parser.parse_args()
    
    download_movielens(args.data_dir)
    analyse_users_preference(rating_dir='./data/ml-100k/u.data',item_dir='./data/ml-100k/u.item')