#!/usr/bin/env python 

"""Class to preprocess mnist dataset using TFRecord 

    Utilzing tensorflow protobuf library, mnist dataset will
    get batched, and shuffled. The new transformed dataset is
    accessable using PreprocessMnist's class methods
    
"""

import os
import sys
import keras 
import numpy as np
import tensorflow as tf
import threading
from typing import Union

Valid_PERCENTAGE_OF_TRAIN = 0.2
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 

rng = np.random.default_rng(seed=42)
(x_train_full, y_train_full), (x_test, y_test) =  keras.datasets.fashion_mnist.load_data()
VALIDATON_SIZE = int(Valid_PERCENTAGE_OF_TRAIN * len(y_train_full))
(x_valid, x_train) = x_train_full[:VALIDATON_SIZE], x_train_full[VALIDATON_SIZE:] 
(y_valid, y_train) = y_train_full[:VALIDATON_SIZE], y_train_full[VALIDATON_SIZE:]

def tfrecords_found()-> bool:
    """Notifies caller whether PreprocessMnist's TFRecords exists  

        If parents directories ..PreprocessMnist/test, ..PreprocessMnist/training, ..PreprocessMnist/validate 
        are not found, they will be created

        
    Args:
        None

        
    Returns:
        A boolean notifyng class that TFRecords don't exist or files are malformed. 
    """
    result = True
    for dir_name in PreprocessMnist.DATASET_TABLE:
        state_path = []
        target_dir = os.path.join(CURR_DIR, f'{__file__[:-3]}', dir_name)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            print("created dir: {}", target_dir)
            result = result and False 
        for entry in os.scandir(target_dir):
            entry_size = os.stat(entry.path).st_size
            if not state_path:
                state_path = [entry_size]
            if len(state_path) and entry_size not in state_path:
                result = result and False 
            state_path[0] = entry_size
        
    return result

def mnist_batch_feature(img : np, label: int) ->tf.train.Example:
    """Create a tf.train.Example protobuf variable representing the mnist features 
    
    
        Args:
            img: 28x28 numpy array of image
            label: Encoded digit value of  img
        
            
        Returns:
            Example protobuf
    """
    feature = { 'feature_image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(img).numpy() ]) ), 'feature_label_64': tf.train.Feature(int64_list=tf.train.Int64List(value=[ label ]))}
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_batch_to_file_helper(batch_id: bool, dataset_name: str, data_x: np.ndarray, data_y: np.ndarray) -> None:
    """Create files to write batch data arrays to file using TFRecord Protobuf format.

    
    Args:
        batch_id: Integer identifier included in filename of batch file
        dataset_name:  String of dataset being evaluated (e.g. "test")
        data_x:  Multtidimensional array of feature data 
        data_y: Multidimensional array of label data
        
    Returns:
        None
    """
    proto_filename = os.path.join(CURR_DIR, f'{__file__[:-3]}', dataset_name,  f'batch_{batch_id}.tfrecord') #  "test.tfrecord")
    with tf.io.TFRecordWriter(proto_filename) as writer:
        for i in range(len(data_x)):
            mnist_proto = mnist_batch_feature(data_x[i], data_y[i])
            mnist_proto_encoded = mnist_proto.SerializeToString()
            writer.write(mnist_proto_encoded)

def stack_dataset(dataset: Union[dict, tuple] ) -> list[np.ndarray, np.array]:
    """Stack dataset
    
    
    Args:
        dataset: iterative or dict housing feature and label samples 
    
    
    Returns:
        Tuple of two arrays

    
    Raises:
        ValueError on invalid dataset type
    """
    result_stateful = []
    for record in dataset:

        if  isinstance(record, dict):
            label = tf.get_static_value(record['feature_label_64']) # read decoded data label
            img = tf.io.parse_tensor( serialized= record['feature_image'], out_type=tf.uint8).numpy() # read decoded data img
        elif isinstance(record, tuple):
            label = tf.get_static_value(record[1]) 
            img = record[0].numpy()[0] 
        else:
            raise ValueError("dataset argument type incorrect ")
        
        if not result_stateful:
            result_stateful += [[img], [label]]
        else:
            result_stateful[0] = np.vstack((result_stateful[0], [img]))
            result_stateful[1] = np.vstack((result_stateful[1], [label]))
    result_stateful[1] = result_stateful[1].reshape(-1)
    return result_stateful[0], result_stateful[1]

class PreprocessMnist:
    """ PreprocessMnist preprocess mnist dataset.

        
        Preprocess involves randomizing the dataset to help reduce bias.

    
    Attributes:
        batch_size: Number of instances per batch. 
        number_of_batches: Number of batches.
        validation_set_size: Number of instances in a validation set
        training_set_size: Number of instances in training set
        test_set_size: Number of instances in test set 
        standardizer: keras layer used to standardize feature data 
    """
    CLASS_LABELS = ["T-shirt/top", "Trouser", "Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    FEATURE_DESC = {'feature_image': tf.io.FixedLenFeature([], tf.string),'feature_label_64': tf.io.FixedLenFeature([], tf.int64, default_value=0)}
    DATASET_TABLE = {'validate' : (x_valid, y_valid), 'training' : (x_train, y_train), 'test' : (x_test, y_test)}
    LIMITED_MODE_DATASET_SIZE = 5000
    
    def __init__(self, load_tf_records: bool=False, limited_mode = True):
        """ Initialize instance
        
        Args:
            load_tf_records: Defines whether to pull and preprocess mnist set. This may produce slower runtime. (Default = False)
            
            limited_mode: Due to non-optimal transformation, process operates in limited mode. Max dataset size is 5000
        """
        self.batch_size = None 
        self.number_of_batches = None  
        self.validation_set_size = VALIDATON_SIZE 
        self.training_set_size = len(y_train)
        self.test_set_size = len(y_test)
        self.standardizer = keras.layers.Normalization()
        
        # rng
        length_percent_25_train = int(0.10 * len(x_train)) 
        rand_indices = rng.integers(low =1, high=len(x_train), size=(length_percent_25_train))
        self.standardizer.adapt(  x_train[rand_indices] )

        if load_tf_records or (not tfrecords_found()):
            for dataset_nanme in PreprocessMnist.DATASET_TABLE:
                x, y = PreprocessMnist.DATASET_TABLE[dataset_nanme]
                if limited_mode:
                    sample_size = PreprocessMnist.LIMITED_MODE_DATASET_SIZE
                    x = x[:sample_size]
                    y = y[:sample_size]
                self.batch_size = int(0.05*len(x))
                y = (lambda y_data:  tf.data.Dataset.from_tensor_slices(y_data).shuffle(buffer_size=5, seed=42).batch(batch_size=self.batch_size, drop_remainder=True)   )(y)
                x = (lambda x_data:  tf.data.Dataset.from_tensor_slices((x_data) ).shuffle(buffer_size=5, seed=42).batch(batch_size=self.batch_size, drop_remainder=True)   )(x)

                self.number_of_batches = len(y)
                concat_xy  = list(x.concatenate(y).as_numpy_iterator())
                for batch_index in range(self.number_of_batches):
                    write_batch_to_file_helper(batch_index, dataset_nanme, concat_xy[batch_index], concat_xy[ batch_index + self.number_of_batches])
        


    def peek_batch(self, dataset_name="training"):
        """ Peek into one of the available dataset_name directories and select a random batch file, analyzing the encoded data
        

            Args:
                dataset_name: Dataset type (e.g. training)
            
                
            Returns:
                List of tensors  
            
                
            Raises:
                FileNotFoundError on file access error 
                ValueError if dataset_name does not exist 
        """
        target_dir = os.path.join(CURR_DIR, f'{__file__[:-3]}', dataset_name)
        target_dir_files = [dir_entry.path for dir_entry in os.scandir(target_dir)]
        target_path = os.path.join(target_dir, f'batch_{rng.integers(low=0, high=len(target_dir_files)) - 1 }.tfrecord')

        if dataset_name not in PreprocessMnist.DATASET_TABLE:
            raise ValueError("name does not exist")
        if not os.path.exists(target_path):
            raise FileNotFoundError("batch file not found")

        files_dataset = tf.data.TFRecordDataset(target_path )
        files_parsed_dataset = files_dataset.map(lambda enc_proto_record: tf.io.parse_single_example(enc_proto_record, PreprocessMnist.FEATURE_DESC) )
        for dataset in files_parsed_dataset.take(2):
            print(dataset)

    def get_data(self, dataset_name= "training"):
        """ Decodes TFRecord files of a dataset identitifed by dataset_name.
            
            
            Function returns a tuple seperating feature and label data.
        

            Args:
                dataset_name: Dataset type


            Returns:
                Tuple with feature and label data. By default, 
                training dataset is returned; the type of dataset
                returned is controlled by dataset_name argument 
            
        """

        batch_dr = os.path.join(CURR_DIR, f'{__file__[:-3]}', dataset_name)
        batch_files = [dir_entry.path for dir_entry in os.scandir(batch_dr)]
        tf_record_dataset = tf.data.TFRecordDataset(batch_files)  
        tf_record_parsed_dataset = tf_record_dataset.map(lambda enc_proto_record: tf.io.parse_single_example(enc_proto_record, PreprocessMnist.FEATURE_DESC), num_parallel_calls=5) # decode protobuf record
        return stack_dataset(tf_record_parsed_dataset)
    
    def load_data(self) -> list[tuple]:
        """Load validation, training, and test datasets
        
        
        Args:
        
        
        Returns:
            Tri-tuple of feature and label data
        
        """
        (train_x, train_y) = self.get_data(dataset_name="training")
        (validate_x, validate_y) = self.get_data(dataset_name="validate")
        (test_x, test_y) = self.get_data(dataset_name="test")

        return [ (train_x, train_y) ,(test_x, test_y) ,  (validate_x, validate_y) ]
    
    def load_dataset(self, dataset_name="training", n_thread_workers=5, n_readers=5, batch_size=1):
        """"Load validation, training, and test datasets using keras dataflow architecture

        
        Args:
            dataset_name: Dataset type
            n_thread_workers: number of async threads 
            n_readers: number of files to
            batch_size: batch size of packets


        Returns:
            Three keras datasets in the following order<train, test, validate>
        """

        def decoder_process(serial_data):
            """Decodes serial data mapped to feature names
            

            Args:
                serial_data: binary string of byyes
                keys: string values used to read dict
            
                
            Returns:
                Tuple of feature and label data
            """
            img = tf.io.parse_tensor( serial_data['feature_image'] ,  out_type=tf.uint8)
            label = (serial_data['feature_label_64'])
            return img , label

        batch_dr = os.path.join(CURR_DIR, f'{__file__[:-3]}', dataset_name)
        batch_filenames = [dir_entry.path for dir_entry in os.scandir(batch_dr)]

        tf_filename_dataset = tf.data.Dataset.list_files(batch_filenames)
        
        dataset = tf_filename_dataset.interleave(
                    cycle_length = n_readers,
                    map_func = lambda unique_dataset_filename: tf.data.TFRecordDataset(unique_dataset_filename)
                                                            .map(lambda protobuf_binary_record: tf.io.parse_single_example(protobuf_binary_record, PreprocessMnist.FEATURE_DESC), num_parallel_calls= n_thread_workers)
                                                            .map(decoder_process, num_parallel_calls=n_thread_workers)
                                                            .shuffle(buffer_size=10)
                                                            .batch(batch_size=batch_size, drop_remainder=False, num_parallel_calls= n_thread_workers)
                                                            .prefetch(1)
        )

        return stack_dataset(dataset)

if __name__ == '__main__':
    inst = PreprocessMnist(True)
    (x, y) = inst.load_dataset()