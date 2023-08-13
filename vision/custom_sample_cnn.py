import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, AUC

class Sample_CNN(tf.Module):

    def __init__(self, num_classes = 2):
        super(Sample_CNN, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu')
        self.maxpool1 = MaxPooling2D((2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.maxpool2 = MaxPooling2D((2, 2))
        self.conv3 = Conv2D(128, (3, 3), activation='relu')
        self.maxpool3 = MaxPooling2D((2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(num_classes, activation='softmax')

    def __call__(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)
    
custom_model = Sample_CNN()

input_data = tf.random.normal((10, 224,224, 3))
output_logits = custom_model(input_data)

print(custom_model)
# tf.keras.utils.plot_model(custom_model, show_shapes=True)