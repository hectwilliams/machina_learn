"""Fetch movie reviews dataset and feed into keras model

    
    Movie reviews datasets are fetched from https://ai.stanford.edu/~amaas/data/sentiment/;
    the model will preprocess and train a binary classificatiion model containing embeddings 
"""

import os
import urllib.request 
import requests
import urllib
import tarfile
import tensorflow as tf 
import keras 
import sys 
import numpy as np 

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
DATASET_LOCAL_DIR = os.path.join(CURR_DIR, __file__[:-3])
DATASET_IMDB_PARENT_DIR = os.path.join(DATASET_LOCAL_DIR, 'aclImdb')
DATA_TRAIN_DIR = os.path.join(DATASET_IMDB_PARENT_DIR, "train")
DATASET_TRAIN_POS_DIR = os.path.join(DATA_TRAIN_DIR,"pos")
DATASET_TRAIN_NEG_DIR = os.path.join(DATA_TRAIN_DIR,"neg")
DATA_TEST_DIR = os.path.join(DATASET_IMDB_PARENT_DIR, "test")
DATASET_TEST_POS_DIR = os.path.join(DATA_TEST_DIR,"pos")
DATASET_TEST_NEG_DIR = os.path.join(DATA_TEST_DIR,"neg")    
TEXT_VECTORIZER_DIMENSION = 4
LOSS = "binary_crossentropy"
OPTIMIZER= "sgd"
METRICS = ["binary_accuracy"]
NUMBER_THREAD_WORKERS = 5
BATCH_SIZE = 1000

def fetch_datasets_url():
    """Using link, download movie_review dataset into movie_review directory 
    

    Args:
    

    Returns:
        None 


    Raises:
        ConnectionError if request fails
    """
    if not requests.head(DATASET_URL).status_code:
        raise ConnectionError()
    if not os.path.exists(DATASET_LOCAL_DIR):
        os.makedirs(name=DATASET_LOCAL_DIR)
    if not [ _  for _ in os.scandir(DATASET_LOCAL_DIR)  ] :
        try:
            imdb_tgz_path = os.path.join(DATASET_LOCAL_DIR, 'imdb.tgz')
            urllib.request.urlretrieve(url=DATASET_URL, filename=imdb_tgz_path)
            imdb_tgz = tarfile.open(imdb_tgz_path)
            imdb_tgz.extractall(path=DATASET_LOCAL_DIR)
            imdb_tgz.close()
        except Exception as e:
            print(f'EXCEPTION: {e}')


def load_datasets():
    """Reads local movie review dataset(s) and preprocess data 
    

    Args:

    
    Returns:
        Three tuple variables ( (training_feat, training_label), (testing_feat, testing_label), (validate_feat, validate_label))

    
    """

    def shuffle_movie_review_files(dataset_name= "train") -> tf.data.Dataset:
        """Shuffles list of pos and neg mmovie review filepaths

        
        Args:
            dataset_name: Leaf directory name where pos and neg reviews are located


        Returns:
            Dataset containing movie review training instances 

            
        Raises:
            FileNotFoundError if dataset directory does not exist 

        """
        if dataset_name == "train":
            pos_dir = os.path.join(DATASET_TRAIN_POS_DIR)
            neg_dir = os.path.join(DATASET_TRAIN_NEG_DIR)
        
        if dataset_name == "test":
            pos_dir = os.path.join(DATASET_TEST_POS_DIR)
            neg_dir = os.path.join(DATASET_TEST_NEG_DIR)
            
        for dir in [pos_dir, neg_dir]:
            if not os.path.exists(dir):
                raise FileNotFoundError(f'{dataset_name} directory not found')
        
        pos_files = [ entry.path for entry in os.scandir(pos_dir) ]
        neg_files = [ entry.path for entry in os.scandir(neg_dir) ]

        pos_dataset_files = tf.data.Dataset.list_files(pos_files)
        neg_dataset_files = tf.data.Dataset.list_files(neg_files)
        
        pos_dataset = pos_dataset_files.map(map_func=lambda filepath:   [tf.io.read_file((filepath)), 1]    , num_parallel_calls=NUMBER_THREAD_WORKERS)
        neg_dataset = neg_dataset_files.map(map_func=lambda filepath:   [tf.io.read_file((filepath)), 0]    , num_parallel_calls=NUMBER_THREAD_WORKERS)
        pos_neg_dataset = pos_dataset.concatenate(neg_dataset)

        return pos_neg_dataset.shuffle(buffer_size= pos_files.__len__() + neg_files.__len__() , seed=42).repeat(4).batch(batch_size=BATCH_SIZE, drop_remainder=True, num_parallel_calls=NUMBER_THREAD_WORKERS)

    options = tf.data.Options()
    options.threading.private_threadpool_size = 0 # system will find optimal

    number_of_batches= int(np.floor( len(os.listdir(DATASET_TRAIN_POS_DIR) + os.listdir(DATASET_TRAIN_NEG_DIR) ) / BATCH_SIZE ))
    train_dataset= shuffle_movie_review_files()
    
    text_vectorization = keras.layers.TextVectorization(standardize= "lower_and_strip_punctuation", split="whitespace",  output_sequence_length=TEXT_VECTORIZER_DIMENSION) 
    text_vectorization.adapt(train_dataset.map(lambda batch_text, batch_label: batch_text, num_parallel_calls=NUMBER_THREAD_WORKERS))

    train_dataset = train_dataset.map(lambda batch_text, batch_label: (text_vectorization(batch_text), batch_label), num_parallel_calls=NUMBER_THREAD_WORKERS) # vectorize batches

    model = keras.models.Sequential(
                [
                    keras.layers.Input(shape=(4,), batch_size=BATCH_SIZE),
                    keras.layers.Dense(300, activation= keras.activations.relu),
                    keras.layers.Dense(200, activation= keras.activations.relu),
                    keras.layers.Dense(100, activation= keras.activations.relu),
                    keras.layers.Dense(1, activation= keras.activations.sigmoid)
                ]   
            )
    keras.utils.plot_model(model, os.path.join(CURR_DIR, f'{__file__[:-3]}' + "flow" + ".png"), show_shapes=True)
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    model.fit(train_dataset, epochs= 4, steps_per_epoch= number_of_batches )

if __name__ == "__main__":
    fetch_datasets_url()
    load_datasets()
