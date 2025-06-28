import os
import pandas as pd
import numpy as np
import time
import logging
from sqlalchemy import create_engine
import urllib.parse

# Konfigurasi logging dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_env_variable(var_name):
    """Mendapatkan environment variable dan memberikan error jika tidak ada."""
    value = os.getenv(var_name)
    if value is None:
        raise ValueError(f"Environment variable '{var_name}' tidak diatur.")
    return value

def create_db_engine():
    """Membuat koneksi engine ke database menggunakan environment variables."""
    db_user = get_env_variable("DB_USER")
    db_password = urllib.parse.quote_plus(get_env_variable("DB_PASSWORD"))
    db_host = get_env_variable("DB_HOST")
    db_port = get_env_variable("DB_PORT")
    db_name = get_env_variable("DB_NAME")
    
    database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    logging.info("Mencoba membuat koneksi ke database...")
    engine = create_engine(database_url)
    logging.info("Koneksi ke database berhasil dibuat.")
    return engine

def process_data(folder_path):
    """Fungsi utama untuk memproses semua file data IMDb."""
    
    # --- Membaca dan membersihkan data dasar ---
    logging.info("Memproses title.basics.tsv...")
    basics_df = pd.read_csv(os.path.join(folder_path, 'title.basics.tsv'), sep='\t', na_values='\\N', low_memory=False)
    movies_df = basics_df[basics_df['titleType'] == 'movie'].copy()
    movies_df = movies_df.drop(columns=['titleType', 'endYear', 'originalTitle'])
    movies_df['startYear'] = pd.to_numeric(movies_df['startYear'], errors='coerce')
    movies_df['runtimeMinutes'] = pd.to_numeric(movies_df['runtimeMinutes'], errors='coerce')
    movies_df.dropna(subset=['startYear', 'runtimeMinutes', 'genres'], inplace=True)
    movies_df['startYear'] = movies_df['startYear'].astype(int)

    logging.info("Memproses title.ratings.tsv...")
    ratings_df = pd.read_csv(os.path.join(folder_path, 'title.ratings.tsv'), sep='\t', na_values='\\N')
    movies_with_ratings_df = pd.merge(movies_df, ratings_df, on='tconst', how='inner')

    logging.info("Memproses title.crew.tsv...")
    crew_df = pd.read_csv(os.path.join(folder_path, 'title.crew.tsv'), sep='\t', na_values='\\N')
    crew_df = crew_df.rename(columns={'directors': 'nconst_director'})
    crew_df = crew_df[['tconst', 'nconst_director']]
    crew_df['nconst_director'] = crew_df['nconst_director'].str.split(',').str[0]
    crew_df.dropna(subset=['nconst_director'], inplace=True)
    
    final_df = pd.merge(movies_with_ratings_df, crew_df, on='tconst', how='inner')

    logging.info("Memproses name.basics.tsv...")
    names_df = pd.read_csv(os.path.join(folder_path, 'name.basics.tsv'), sep='\t', na_values='\\N')
    names_df_subset = names_df[['nconst', 'primaryName']].rename(columns={'nconst': 'nconst_director', 'primaryName': 'directorName'})
    final_df = pd.merge(final_df, names_df_subset, on='nconst_director', how='left')
    final_df = final_df.drop(columns=['nconst_director'])
    
    # --- Proses file besar dengan chunking ---
    logging.info("Memproses title.principals.tsv dengan chunking...")
    movie_tconsts_set = set(final_df['tconst'].unique())
    chunk_iterator = pd.read_csv(
        os.path.join(folder_path, 'title.principals.tsv'), sep='\t', na_values='\\N',
        usecols=['tconst', 'ordering', 'nconst', 'category'], dtype={'category': 'category'}, chunksize=1000000
    )
    list_of_processed_chunks = []
    for chunk in chunk_iterator:
        filtered_chunk = chunk[chunk['category'].isin(['actor', 'actress'])]
        relevant_chunk = filtered_chunk[filtered_chunk['tconst'].isin(movie_tconsts_set)]
        if not relevant_chunk.empty:
            list_of_processed_chunks.append(relevant_chunk)
    actors_df = pd.concat(list_of_processed_chunks, ignore_index=True)
    
    names_original_df = names_df[['nconst', 'primaryName']]
    actors_with_names_df = pd.merge(actors_df, names_original_df, on='nconst', how='left')
    actors_with_names_df = actors_with_names_df.sort_values(by=['tconst', 'ordering'])
    actor_list_df = actors_with_names_df.groupby('tconst')['primaryName'].apply(lambda x: list(x.head(5))).reset_index()
    actor_list_df = actor_list_df.rename(columns={'primaryName': 'actors'})
    final_df = pd.merge(final_df, actor_list_df, on='tconst', how='left')
    
    logging.info("✅ Proses cleaning dan merging data film selesai.")
    return final_df

def main():
    """Fungsi orkestrasi utama."""
    try:
        # Dapatkan path folder data dari environment variable
        folder_path = get_env_variable("DATA_PATH")
        
        # Proses data untuk menghasilkan DataFrame final
        final_df = process_data(folder_path)
        
        # Di sini Anda bisa menambahkan logika untuk mengubah final_df menjadi star schema
        # dan memuatnya ke database, atau memuatnya sebagai tabel staging.
        # Untuk saat ini, kita akan simpan sebagai contoh.
        
        output_path = os.path.join(folder_path, 'cleaned_imdb_movies_batch.parquet')
        final_df.to_parquet(output_path, index=False)
        logging.info(f"Proses batch selesai. File bersih disimpan di: {output_path}")

    except Exception as e:
        logging.error(f"Proses batch gagal: {e}")

if __name__ == "__main__":
    main()