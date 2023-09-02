import sys
sys.path.append("../")

import tensorflow as tf
import tensorflow_datasets as tfds

from functional_style_sample_cnn import func_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, AUC


train_dataset= tfds.load('cifar10', split="train", with_info=False)
test_dataset= tfds.load('cifar10', split="test", with_info=False)

func_model.compile(  optimizer=SGD(learning_rate = 0.0001,),
                    loss = BinaryCrossentropy(from_logits=False), 
                    metrics=[AUC(),
                             Accuracy()]
                )


func_model.fit(train_dataset, epochs=10, validation_data=test_dataset)