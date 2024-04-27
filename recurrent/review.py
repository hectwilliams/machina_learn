"""Sentiment analysis using imdb movie review
"""
import tensorflow_datasets as tfds
import tensorflow as tf 
from collections import Counter 
import keras 
import sys 
import numpy as np 

DATASET_NAME = "imdb_reviews"
BATCH_SIZE = 32
REGEX_BREAK_TAGS = b"<br\\s*/?>"
REGEX_KEEP_LETTER_AND_QUOTE = b"[^a-zA-Z']"
REVIEW_WINDOW_SIZE = 300
SPACE = b" "
N_OOV_BUCKETS = 1000 
EMBEDDING_DIMENSIONS = 128
GRU_UNITS = 128 

def preprocess(x_batch, y_batch):
    """Transform x_batch containing movie reviews 

    Function removes punctuations, and html tags from movie 
    reviewns and splits are valid string words.
    

    Args:
        x_batch: feature data batch, collection of tensor strings 
        y_batch: label data batch, collection of tensor ints
        
    
    Returns:
        Tuple containing processed x_batch and y_batch
    """
    x_batch = tf.strings.substr(x_batch, 0, REVIEW_WINDOW_SIZE)
    x_batch = tf.strings.regex_replace(x_batch, REGEX_BREAK_TAGS, SPACE)
    x_batch = tf.strings.regex_replace(x_batch, REGEX_KEEP_LETTER_AND_QUOTE, SPACE)
    x_batch = tf.strings.split(x_batch) 
    x_batch = x_batch.to_tensor(default_value=b"<pad>") # ragged tensors to tensor
    return x_batch, y_batch

def count_words(dataset: tf.data.Dataset, vocab: Counter):
    """Counter object (i.e. vocab) structure tracks/counts the occurence of words in dataset.

    
    Args:
        dataset: Dataset containing batched movie reviews encoded into words
        vocab: Counter object 
    """
    for x_batch, _ in dataset:
        for review_words in x_batch:
            vocab.update(list(review_words.numpy()))

def create_lut(vocab: Counter) -> tf.lookup.StaticVocabularyTable:
    """Generate lookup table for words contained in vocab (i.e. word occurence count table)
    
    
    Args:
        vocab: _
    
    
    Returns:
        LUT mapping each word to a unique ID
    """
    words_tf = tf.constant([word for word, count in vocab.items()])
    words_id_tf = tf.range(len(words_tf), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words_tf, words_id_tf)
    return tf.lookup.StaticVocabularyTable(vocab_init, N_OOV_BUCKETS)

def encode_words(x_batch, y_batch):
    """Encode batched movie review data containing words
    

    Args:
        x_batch: _
        y_batch: _
    
    Returns:
        Dataset containing encoded batch
    """

vocab_table = Counter()
datasets, info = tfds.load(DATASET_NAME, as_supervised=True, with_info=True)
train_size = info.splits["train"].num_examples # train-size 25000
test_size = info.splits["test"].num_examples  #test-size 25000   

train_ds = datasets.get('train').batch(BATCH_SIZE).map(preprocess)
count_words(train_ds, vocab_table)
lut_tf = create_lut(vocab_table)
train_ds = train_ds.map(lambda x_batch, y_batch: (lut_tf.lookup(x_batch), y_batch)).prefetch(1)

model = keras.models.Sequential(
    [
        keras.layers.Embedding(input_dim=int(lut_tf.size()), output_dim=EMBEDDING_DIMENSIONS, input_shape=[None]),
        keras.layers.GRU(GRU_UNITS, return_sequences=True),
        keras.layers.GRU(GRU_UNITS),
        keras.layers.Dense(1, activation=keras.activations.sigmoid)
    ]
)
model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
model.fit(train_ds)