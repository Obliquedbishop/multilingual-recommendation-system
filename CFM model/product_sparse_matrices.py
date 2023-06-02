import multiprocessing as mp
import numpy as np
import pandas as pd
from functools import lru_cache
import os
import dill


count = 0

def get_indices(x, products_df, total_len):
    global count
    if count % 10000 == 0:
        print(round((count/total_len) * 100, 2), ' % Done.')
    count += 1
    return list(np.where(products_df['id'].isin(x)))

def process_session_chunk(sess_df):
    items_viewed = sess_df['prev_items'].to_numpy()
    max_length = sess_df['prev_items'].apply(lambda sess: len(sess)).max()
    items_viewed = items_viewed.astype(f'<U{max_length}')
    stripped_items = np.char.strip(np.char.strip(items_viewed, r'[]'))
    replaced_items = np.char.replace(np.char.replace(stripped_items, '\'', ''), '\n', '')
    splitted_items = np.char.split(replaced_items)
    return splitted_items


def generate_sparse_matrix(user_session_df, products_df, locale):
    pool = mp.Pool()
    processed_sess_chunks = pool.map(process_session_chunk, np.array_split(user_session_df, 10))
    pool.close()
    pool.join()
    combined_splitted_items = np.concatenate(processed_sess_chunks)
    indices = np.array(list(map(lambda x: get_indices(x, products_df, combined_splitted_items.shape[0]), combined_splitted_items)))
    np.save(r'C:\Users\Shreyansh\Desktop\dev\VLG_open_project\multilingual_recommendation_system\CFM model\sparse_indices_'+locale+r'.npy', indices)
    print("**************************************************************************************************************")


@lru_cache(maxsize=1)
def read_product_data(train_data_dir):
    return pd.read_csv(os.path.join(train_data_dir, 'products_train.csv'))

@lru_cache(maxsize=1)
def read_train_data(train_data_dir):
    return pd.read_csv(os.path.join(train_data_dir, 'sessions_train.csv'))

@lru_cache(maxsize=3)
def read_test_data(task, test_data_dir):
    return pd.read_csv(os.path.join(test_data_dir, f'sessions_test_{task}.csv'))


if __name__ == '__main__':

    train_data_dir = r'C:\Users\Shreyansh\Desktop\dev\VLG_open_project\multilingual_recommendation_system\CFM model\train_data'
    test_data_dir = r'C:\Users\Shreyansh\Desktop\dev\VLG_open_project\multilingual_recommendation_system\CFM model\test_data'

    user_session_df = read_train_data(train_data_dir)
    products_df = read_product_data(train_data_dir)
    unique_locales = products_df['locale'].unique()
    for locale in unique_locales:
        if locale == 'IT':
            continue
        print(f"Starting with locale: {locale}")
        user_session_df = user_session_df.query(f'locale=="{locale}"')
        products_df = products_df.query(f'locale=="{locale}"')

        generate_sparse_matrix(user_session_df, products_df, locale)

