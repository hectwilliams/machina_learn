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
RNN_UNIT = 128 
DEBUG_ON = True
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

def next_char(text: str, temperature: float=1.0, tokenizer:  tf.keras.preprocessing.text.Tokenizer = None, model: keras.models.Sequential = None, char_table_size: int = None)-> str:
    """Predicts next character using RNN
    
    
    Args:
        text: 1-Dimensional text
        temperature: Float value between 0 and 1 (inclusive)
        tokenizer: tokenizer
        model: Keras model
        char_table_size: Integer value represents number of unique token elements
        
    
        
    Raises:
        ValueError if model or char_table_size are None

    
    Returns:
        Single character
    """
    if char_table_size == None or model == None or tokenizer == None:
        raise ValueError("None value received")
    
    print("DEBUG:\t {}".format(text))
    
    x_new = preprocess_char_rnn(tokenizer, [text], char_table_size)
    next_char_prob = model.predict(x_new)[0, -1:, :] # probability of character token
    rescaled_logits =  tf.math.log(next_char_prob) / temperature #  log distribution
    id_zero_based = tf.random.categorical(rescaled_logits, num_samples=1) 
    class_id_ones_based = (id_zero_based + 1).numpy()
    return tokenizer.sequences_to_texts(class_id_ones_based)[0]

def complete_text(text: str , n_chars: int = 20, temperature:float= 1.0,  tokenizer:  tf.keras.preprocessing.text.Tokenizer = None, model: keras.models.Sequential = None, char_table_size: int = None)->str:
    """Repeatedly calls next_char() to generate sequence
    
    
    Args:
        text: string of zero or more characters 
        n_chars: number of characters to predict
        temperature: Float value between 0 and 1 (inclusive)
        tokenizer: tokenizer
        model: Keras model
        char_table_size: Integer value represents number of unique token elements
    
        

    Returns
        Sentence string
    """
    for _ in range(n_chars):
        text += next_char(text, temperature, tokenizer, model, char_table_size)
    return text 

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
    rnn_units = 10 if DEBUG_ON else RNN_UNIT
    model = keras.models.Sequential(
        [
            keras.layers.GRU( rnn_units, return_sequences=True, input_shape=(None, number_of_unique_chars), dropout=0.2, recurrent_dropout=0.2),
            keras.layers.GRU( rnn_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            keras.layers.TimeDistributed(
                keras.layers.Dense(number_of_unique_chars, activation=keras.activations.softmax)
            )
        ]
    )
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
    history = model.fit(train_dataset)

    s = "hi my nam"
    s = complete_text(s,10, 1.0, tokenizer, model, number_of_unique_chars)       

print(f'ELASPED TIME - {default_timer() -  start}')