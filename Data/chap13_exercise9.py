#!/usr/bin/env python 

"""
    Exercise utilizing TFRecord and Standardization

    Usage: chap13_exercise9.py 
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras 
import zlib
import sys

def dir_check(folder_names, curr_dir):
    '''
        Check for validate, training, and test direcories for TFRecord files
        
        If not found, create folders for script  

    '''
    for name in folder_names:
        new_path = os.path.join(curr_dir, f'{__file__[:-3]}',name )
        if not os.path.isdir(new_path):
            os.makedirs(new_path)

def mnist_batch_feature(img , label)  :
    feature = {
        'feature_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(img).numpy() ]) ),
        'feature_label_64': tf.train.Feature(int64_list=tf.train.Int64List(value=[ label ])),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

wk_dir = os.path.join(os.path.dirname(os.path.realpath(__file__))) 

# fashion mnst ()

class_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

feature_desc = {
    'feature_image': tf.io.FixedLenFeature([], tf.string),
    'feature_label_64': tf.io.FixedLenFeature([], tf.int64, default_value=0),
}


fashion_mnist = keras.datasets.fashion_mnist

(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
percentage_of_train = 0.2
valid_length = int(percentage_of_train * len(y_train_full))
(x_valid, x_train) = x_train_full[:valid_length], x_train_full[valid_length:]
(y_valid, y_train) = y_train_full[:valid_length], y_train_full[valid_length:]


# Preprocessing data and writing to file 

dataset_table = {
    'validate' : (x_valid, y_valid),
    'training' : (x_train, y_train),
    'test' : (x_test, y_test),
}

dir_check(dataset_table, wk_dir)

for dataset_name in dataset_table:
    x, y = dataset_table[dataset_name]
    print(dataset_name)

    size = len(x)
    batch_size = int(0.10 * size)
    print("DEBUG--BATCH_SIZE {} {}".format(batch_size, size))

    #  convert to tensor slices ( both x and y)

    y = (lambda y_data:  tf.data.Dataset.from_tensor_slices(y_data))(y)
    x = (lambda x_data:  tf.data.Dataset.from_tensor_slices(x_data))(x)

    # shuffle 

    y = (lambda y_tf_data: y_tf_data.shuffle(buffer_size=20, seed=42))(y)
    x = (lambda x_tf_data: x_tf_data.shuffle(buffer_size=20, seed=42))(x)
 
    # batch  [ [batch1]. [batch2].[] ]

    y = (lambda y_tf_data: y_tf_data.batch(batch_size=batch_size, drop_remainder=True))(y)
    x = (lambda x_tf_data: x_tf_data.batch(batch_size=batch_size, drop_remainder=True))(x)
    number_of_batches = len(x)
    x_batches = list(x.as_numpy_iterator())
    y_batches = list(y.as_numpy_iterator())

    # encode batch data and write file 

    for batch_idx in range(number_of_batches):
        proto_filename = os.path.join(wk_dir, f'{__file__[:-3]}', dataset_name,  f'batch_{batch_idx}.tfrecord') #  "test.tfrecord")
        with tf.io.TFRecordWriter(proto_filename) as writer:
            x_batch_data = x_batches[batch_idx]
            y_batch_data = y_batches[batch_idx]
            x_tf = tf.convert_to_tensor(x_batch_data)
            y_tf = tf.convert_to_tensor(y_batch_data)
            y_data = tf.data.Dataset.from_tensor_slices( y_batches[batch_idx]) 
            x_data = tf.data.Dataset.from_tensor_slices( x_batches[batch_idx]) 
            mnist_proto = mnist_batch_feature(x_data.__iter__().get_next_as_optional().get_value(), y_data.__iter__().get_next_as_optional().get_value())
            encoded_mnist_proto = mnist_proto.SerializeToString()

            for _ in range(batch_size):
                writer.write(encoded_mnist_proto)

# Reading from file + preprocessing + training 

for dataset_name in dataset_table:
    directory = os.path.join(wk_dir, f'{__file__[:-3]}', dataset_name )
    filenames = [dir_entry.path for dir_entry in os.scandir(directory)]
    
    filepath_dataset = tf.data.Dataset.list_files(filenames, seed=42)
    # Grab 6 files at a time, grabing a single line(record)
    # interleave_dataset = filepath_dataset.interleave(cycle_length=5, map_func=lambda fpath: tf.data.TextLineDataset(fpath))
    files_dataset = tf.data.TFRecordDataset(filenames[:1])
    files_parsed_dataset = files_dataset.map(lambda enc_proto_record: tf.io.parse_single_example(enc_proto_record, feature_desc) )
    
    for decoded_proto_dataset in files_parsed_dataset:
        label = decoded_proto_dataset['feature_label_64']
        img_binary = decoded_proto_dataset['feature_image']
        img_unsigned = tf.io.parse_tensor( serialized= img_binary, out_type=tf.uint8)
        img_unsigned = img_unsigned.numpy()
        print(label)
    
    # for line in interleave_dataset.take(1):
    #     decoded_proto = tf.train.Example.FromString(line.numpy())
    #     print(decoded_proto)

    break

    