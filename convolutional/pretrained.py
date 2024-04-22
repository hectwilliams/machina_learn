"""Using Pretrained Model ResNet-50
"""
import os
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt 
import keras
import tensorflow_datasets as tfds
from timeit import default_timer
import sys 
from functools import reduce 

XCEPTION_MODEL_IMAGE_SIZE_N = 224
BATCH_SIZE = 1 
NUM_INPUTS_TO_RESNET = 3
LOSS = "sparse_categorical_crossentropy"
METRICS = [keras.metrics.Accuracy()] 
OPTIMIZER = keras.optimizers.SGD(learning_rate=0.001) 
CURR_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__))) 
FLOAT_FORMATTER = "{:.3f}".format
MODEL_PATH =  os.path.join(CURR_DIR, __file__[:-3] + ".keras" )

def preprocess_xception(image, label):
    """Resize and preprocess images


    Args:
        image: _
        label: _
    
    
    Return:
        Two element tuple containing processed image and label, respectively
    """
    resized_image = tf.image.resize(image, [XCEPTION_MODEL_IMAGE_SIZE_N, XCEPTION_MODEL_IMAGE_SIZE_N])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

def preprocess_resnet(image, label):
    """Denormalize (re-scale) pixel values 
    
    
    Args:
        image: _
        label: _
    
    
    Returns:
        ...
    """
    resized_image = tf.image.resize(image, [XCEPTION_MODEL_IMAGE_SIZE_N, XCEPTION_MODEL_IMAGE_SIZE_N])
    return resized_image, label

# format float 
np.set_printoptions(formatter={'float_kind':FLOAT_FORMATTER})

# Fetch Data 
dataset, info = tfds.load("tf_flowers", as_supervised=True, with_info=True)
dataset_size = info.splits["train"].num_examples
test, valid, train = tfds.load(name="tf_flowers", split=["train[:10%]", "train[10%:25%]", "train[25%:]"], as_supervised=True)

def my_reduce(accumulator , data ,):
    if  not accumulator :
        return [ np.c_[ data[0] ]  , np.c_[data[1][0] ]  ]
    else:
        return [np.vstack( (accumulator[0], data[0]) ) , np.vstack( (accumulator[1], data[1])  ) ]

def res_50():
   
    test_dataset = test.map(preprocess_resnet).batch(BATCH_SIZE).prefetch(1)
    test_dataset_to_list = list(test_dataset.as_numpy_iterator())
    images, labels = reduce(my_reduce, test_dataset_to_list[:NUM_INPUTS_TO_RESNET] , [] )
    fig, axes = plt.subplots(nrows= NUM_INPUTS_TO_RESNET, ncols=2 )

    base_model = keras.applications.resnet50.ResNet50(weights="imagenet")
    images = keras.applications.resnet50.preprocess_input(images)

    # freeze resnet model 
    for layer in base_model.layers:
        layer.trainable = False 

    # keras model 
    model = keras.models.Sequential()
    model.add( keras.layers.Input(shape=(224,224,3), batch_size=BATCH_SIZE) )
    model.add(base_model)
    model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)
    model.summary()

    # prediction 
    predict_proba = model.predict(images)
    top_k = keras.applications.resnet50.decode_predictions(predict_proba, top=5)
    for img_idx in range(len(images)):
        text = ""
        for class_id , name, prob in top_k[img_idx]:
            text += f'{name}-{prob:.2f}%\n'
        for i in range(2):
            axes[img_idx,i].set_xticks([])        
            axes[img_idx,i].set_yticks([])        
            axes[img_idx,i].axis('off')
            if i == 0:
                axes[img_idx,0].imshow(images[img_idx])
            else:
                axes[img_idx,1].text(0,1,text, {'size': 6, 'verticalalignment':'top'})
    
    fig.suptitle( f'{"Images":<15}{"Prediction Probability":>15}', fontsize=10)
    keras.utils.plot_model(model, os.path.join(CURR_DIR, __file__[:-3] + '_res50.png'), show_shapes=True)
    plt.show()

def xcep(retrain= False):

   # keras model 
    train_ds = train.map(preprocess_xception).batch(BATCH_SIZE).prefetch(1)
    train_ds_to_list = list(train_ds.as_numpy_iterator())
    images, labels = reduce(my_reduce, train_ds_to_list[:NUM_INPUTS_TO_RESNET] , [] )
    fig, axes = plt.subplots(nrows= NUM_INPUTS_TO_RESNET, ncols=2 )
    xception_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
    feature_names = info.features["label"].names
    n_classes = info.features["label"].num_classes

    if not os.path.exists(MODEL_PATH) or retrain:

        # base model layers
        
        # new layers ( to be trained )
        avg = keras.layers.GlobalAveragePooling2D()(xception_model.output)
        output = keras.layers.Dense(n_classes, activation="softmax")(avg)
        model = keras.Model(inputs=xception_model.input, outputs=output)
        
        # freeze base model weights
        for layer in xception_model.layers:
            layer.trainable = False 
        
        print("DEBUG: freezed and train")
        # train new layers 
        optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9, decay=0.01)
        model.compile(loss=LOSS, optimizer=optimizer, metrics=METRICS)
        history = model.fit(train_ds)
        
        print("DEBUG: unfreezed")
        # unfreeze xception model
        for layer in xception_model.layers:
            layer.trainable =  True
        
        print("DEBUG: train all")
        # train all layers 
        optimizer = keras.optimizers.SGD(learning_rate=0.02, momentum=0.9, decay=0.001)
        model.compile(loss=LOSS, optimizer=optimizer, metrics=METRICS)
        history = model.fit(train_ds)
        
        print("DEBUG: train model")
        model.save(MODEL_PATH)

    else:
        model = keras.models.load_model(MODEL_PATH)
    predict_proba = model.predict(images)
    predict_proba_best_idx = np.argmax(predict_proba, 1)
    for i in range(NUM_INPUTS_TO_RESNET):
        axes[i, 0].imshow(images[i])
        axes[i, 0].axis('off')
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].text(0,0, f'Predicted: -  {feature_names[predict_proba_best_idx[i]] }\n Expected - { feature_names[ labels[i][0] ] }')
        axes[i, 1].axis('off')
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
 
    plt.savefig(  os.path.join(CURR_DIR, __file__ [:-3] + "_erroneous_guess" + '.png')   )
    plt.show()
    
if __name__ == "__main__":
    started = default_timer()
    xcep(False)
    print( f'ELASPED RUN TIME: \t{default_timer() - started}\n' )
