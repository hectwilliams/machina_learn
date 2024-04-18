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
import time 
from timeit import default_timer

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
TEXT_VECTORIZER_DIMENSION = 1
LOSS = "binary_crossentropy"
OPTIMIZER= "sgd"
METRICS = ["accuracy"] # binary_accuracy
NUMBER_THREAD_WORKERS = 6 
BATCH_SIZE = 1000
EMBEDDING_DIMENSION = 2
TEXT_VECTORIZER_DIMENSION_EMBEDDING = 1
ONE = 1
ZERO = 0
REVIEWS_IN_DATASET = 25000
VALIDATION_SIZE = 15000
TEST_SIZE = 10000

options = tf.data.Options()
options.threading.private_threadpool_size = 0 # system will find optimal

text_vec_map_to_integer = keras.layers.TextVectorization(standardize= "lower_and_strip_punctuation", split="whitespace",  output_sequence_length=TEXT_VECTORIZER_DIMENSION, output_mode='int') 
text_vec_map_to_integer_train = keras.layers.TextVectorization(standardize= "lower_and_strip_punctuation", split="whitespace",  output_sequence_length=TEXT_VECTORIZER_DIMENSION, output_mode='int') 
text_vec_map_to_integer_test = keras.layers.TextVectorization(standardize= "lower_and_strip_punctuation", split="whitespace",  output_sequence_length=TEXT_VECTORIZER_DIMENSION, output_mode='int') 

text_vect_map_to_indices = keras.layers.TextVectorization(standardize= "lower_and_strip_punctuation", split="whitespace", output_mode='int') 
text_vect_map_to_indices_train = keras.layers.TextVectorization(standardize= "lower_and_strip_punctuation", split="whitespace", output_mode='int') 
text_vect_map_to_indices_test = keras.layers.TextVectorization(standardize= "lower_and_strip_punctuation", split="whitespace", output_mode='int') 

number_of_batches = 1
embedding_layer =  {}
embedding_layer_train = {}
embedding_layer_test = {}

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
    def preprocess_embeddings(batch_review_vectors, batch_of_indices, batch_of_labels):
        """Compute embeddings for batch of indices

        
        Args:
            batch_review_vectors: _
            batch_of_indices: Vectoerized words mapping to vocabulary
            batch_of_labels: _

            
        Returns:
            Three element tuple. First and third element are passthroughs.
            The second element will undergo embedding averaging of size  BATCH_SIZE x EMBEDDING_DIMENSION
        """
        embedds = embedding_layer(batch_of_indices)
        embedds_mean = tf.cast(tf.reduce_mean(embedds, axis=1), tf.float64)
        
        indices_bool_cast = tf.cast(batch_of_indices , tf.bool)
        indices_bin_digit_cast =  tf.cast(indices_bool_cast ,  tf.float64)
        indices_length = tf.reduce_sum(indices_bin_digit_cast, axis=1 , keepdims=True)
        sqrt_reduce_indices = tf.sqrt(indices_length)
        rscaled_mean_embedding  = tf.math.multiply(embedds_mean, sqrt_reduce_indices) # embedds_mean * sqrt_reduce_indices_repeat_stack
        return (batch_review_vectors, rscaled_mean_embedding), (batch_of_labels)
    
    def preprocess_vectorization(batch_of_reviews, batch_of_labels):
        """Vectorization of movie reviews

        
        Args:
            review: batch of reviews
            label: batch of single digit binary choices like(1) or dislike(0)

            
        Returns:
            Three element tuple. First element is batch of vectorized reviews to single 1D location.
            The second element is vectorized batch of reviews; each review is converted to 
            a array of indices mapping to word in a vocabulary. Last element 
            is a batch of binary choices for a corresponding review
        """
        
        return   text_vec_map_to_integer(batch_of_reviews), text_vect_map_to_indices(batch_of_reviews), batch_of_labels
    
    def get_datasets(dataset_name= "train") -> tf.data.Dataset:
        """Shuffles list of pos and neg mmovie review filepaths

        
        Args:
            dataset_name: Leaf directory name where pos and neg reviews are located


        Returns:
            Dataset containing movie review training instances shuffled

            
        Raises:
            FileNotFoundError if dataset directory does not exist 
            ValueError if dataset_name is other then test or train

        """
        def files_for_dataset(name = "train") :
            """Return files for individual dataset


            Args:
                name: Name of dataset

            
            Returns:
                Array of filename strings
            
                
            Raises:
                ValueError if dataset_name is other then test or train
            
            """

            global number_of_batches
            
            pos_dir = ""
            neg_dir = ""
            
            if name not in ["train", "test"]:
                raise ValueError()
            
            if dataset_name == "train":
                pos_dir = os.path.join(DATASET_TRAIN_POS_DIR)
                neg_dir = os.path.join(DATASET_TRAIN_NEG_DIR)
                number_of_batches = int(np.floor( len(os.listdir(DATASET_TRAIN_POS_DIR) + os.listdir(DATASET_TRAIN_NEG_DIR) ) / BATCH_SIZE ))
            elif dataset_name == "test":
                pos_dir = os.path.join(DATASET_TEST_POS_DIR)
                neg_dir = os.path.join(DATASET_TEST_NEG_DIR)
                number_of_batches = int(np.floor( len(os.listdir(DATASET_TEST_POS_DIR) + os.listdir(DATASET_TEST_NEG_DIR) ) / BATCH_SIZE ))
            else:
                raise ValueError("Only test or train values are permitted")
                
            for dir in [pos_dir, neg_dir]:
                if not os.path.exists(dir):
                    raise FileNotFoundError(f'{dataset_name} directory not found')
        
            pos_files = [ entry.path for entry in os.scandir(pos_dir) ]
            neg_files = [ entry.path for entry in os.scandir(neg_dir) ]

            return     neg_files, pos_files   
        
        def get_dataset(n_files, p_files):
            """Using files Compute Dataset
            
            
            Args:
                n_files: _
                p_files: _
            
            
            Returns:
                Dataset
            """

            n_dataset_files = tf.data.Dataset.list_files(n_files)
            p_dataset_files = tf.data.Dataset.list_files(p_files)
            n_dataset = n_dataset_files.map(map_func=lambda filepath: [tf.io.read_file((filepath)), ZERO], num_parallel_calls=NUMBER_THREAD_WORKERS)
            p_dataset = p_dataset_files.map(map_func=lambda filepath: [tf.io.read_file((filepath)), ONE], num_parallel_calls=NUMBER_THREAD_WORKERS)
            return p_dataset.concatenate(n_dataset)
        
        n_files, p_files = files_for_dataset("train")
        train_dataset = get_dataset(n_files, p_files)
        
        n_files, p_files = files_for_dataset("test")
        test_dataset = get_dataset(n_files, p_files)
        

        train_dataset = train_dataset.shuffle(buffer_size= len(train_dataset), seed=42).repeat(1).batch(batch_size=BATCH_SIZE, drop_remainder=True, num_parallel_calls=NUMBER_THREAD_WORKERS)
        test_dataset = test_dataset.shuffle(buffer_size= len(test_dataset), seed=42).repeat(1)
        
        validation_dataset = test_dataset.take(VALIDATION_SIZE).batch(batch_size=BATCH_SIZE, drop_remainder=True, num_parallel_calls=NUMBER_THREAD_WORKERS)
        test_dataset = test_dataset.skip(VALIDATION_SIZE).batch(batch_size=BATCH_SIZE, drop_remainder=True, num_parallel_calls=NUMBER_THREAD_WORKERS)
        
        return train_dataset, test_dataset, validation_dataset


    def preprocess_vectorization_double(train_d, test_d):
        """Vectorization of movie reviews

        
        Args:
            review: batch of reviews
            label: batch of single digit binary choices like(1) or dislike(0)

            
        Returns:
            Three element tuple. First element is batch of vectorized reviews to single 1D location.
            The second element is vectorized batch of reviews; each review is converted to 
            a array of indices mapping to word in a vocabulary. Last element 
            is a batch of binary choices for a corresponding review
        """

        
        return train_d, test_d
        
        return   text_vec_map_to_integer(batch_of_reviews), text_vect_map_to_indices(batch_of_reviews), batch_of_labels

    global embedding_layer_train 
    global embedding_layer_test 
    global embedding_layer

    train_dataset, test_dataset, valid_dataset = get_datasets()
    
 
    
if __name__ == "__main__":
    started = default_timer()
    fetch_datasets_url()
    load_datasets()
    ended = default_timer()
    msg = f'ELASPED RUN TIME: {ended - started}\n'
    print(msg)
