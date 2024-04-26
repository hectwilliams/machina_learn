"""Generate Shakespearean Text Using Character RNN

    Model will train a RNN to predict the next character in a sentence 
"""

from timeit import default_timer
import tensorflow as tf 
import keras 
import numpy as np 
import sys 

SHAKESPEARE_URL = "https://homl.info/shakespeare"
SHAKESPEARE_FILENAME = "shakespeare.txt"
CHAR_ENCODING = True
PERCENT_OF_TEXT_FOR_TRAIN = 0.9
BATCH_SIZE = 32 

start = default_timer()
filepath = keras.utils.get_file(SHAKESPEARE_FILENAME, SHAKESPEARE_URL)

def preprocess_char_rnn(tokenizer, texts, num_of_chars):
    """Tokenize string
    
    
    Args:
        tokenizer: String to integer token transformer
        texts: One dimensional array containing string
        
    
    Returns:
       One-hot encoded text
    """

    x = np.array(tokenizer.texts_to_sequences(texts)) -1
    return tf.one_hot(x, num_of_chars)

with open(filepath) as f:
    shakespeare_text = f.read() 

    # encode every character into an integer
    tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level = CHAR_ENCODING)
    tokenizer.fit_on_texts(shakespeare_text)
    
    # distinct character count 
    number_of_unique_chars = len(tokenizer.word_index)
    number_of_chars = tokenizer.document_count

    # encode full text 
    encoded_text = np.array(tokenizer.texts_to_sequences([shakespeare_text])) - 1
    [encoded_text] = encoded_text

    # split dataset
    train_size = int(np.floor(PERCENT_OF_TEXT_FOR_TRAIN * number_of_chars))
    dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
    train_dataset, test_dataset = dataset.take(train_size), dataset.skip(train_size)

    # dataset of window datasets (windows shifted by one time unit in future)
    n_steps = 100
    window_length = n_steps + 1
    train_dataset = train_dataset.window(size= window_length, shift= 1, drop_remainder= True)
    train_dataset = train_dataset.flat_map(lambda window_ds: window_ds.batch(batch_size= window_length, drop_remainder= True))

    # batch dataset 
    train_dataset = train_dataset.shuffle(buffer_size=10000).batch(BATCH_SIZE)

    # batch to feature_data and label batch
    train_dataset = train_dataset.map(lambda batch_of_windows: ( batch_of_windows[:, :-1], batch_of_windows[:, 1:] ) )

    # one hot the train_feature data 
    train_dataset = train_dataset.map(lambda x_batch, y_batch: ( tf.one_hot(x_batch, depth=number_of_unique_chars) , y_batch) )

    # model 
    model = keras.models.Sequential(
        [
            keras.layers.GRU(128, return_sequences=True, input_shape=(None, number_of_unique_chars), dropout=0.2, recurrent_dropout=0.2),
            keras.layers.GRU(128, return_sequences=True, input_shape=(None, number_of_unique_chars), dropout=0.2, recurrent_dropout=0.2),
            keras.layers.TimeDistributed(keras.layers.Dense(number_of_unique_chars, activation=keras.activations.softmax))
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    
    # prdiction 
    some_string = ["How are yo"]
    some_string_preprocessed = preprocess_char_rnn(tokenizer, some_string, number_of_unique_chars)

    some_prediction_zero_based = model.predict(some_string_preprocessed)
    some_prediction_zero_based.shape  # (1, 10, 39)
    some_prediction_zero_based_classes_indices = (np.argmax(some_prediction_zero_based, axis=1)).astype(np.int32) 
    
    predicted_classes_idx = (some_prediction_zero_based_classes_indices) + 1 # changes to ones_based
    char_keys = (np.array(list(tokenizer.word_index.keys())))
    predicted_classes = char_keys[predicted_classes_idx] 
    predicted_next_char = predicted_classes[0][-1]
    print(f'Predict char after {some_string} -> [{predicted_next_char}]' )

print(f'ELASPED TIME - {default_timer() -  start}')