import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])

print(model.summary())

"""
    Batch Normaliztion her bir feature icin 4 yeni learnable parameter eklemis
    784 * 4 = 3136
    
    Bir detay daha: 
        dedigimiz gibi mean ve std burda moving average dir ve backpropagation da hesapalnmaz bu yuuzden 
    
    Total params: 271,346
    Trainable params: 268,978
    Non-trainable params: 2,368
    
    2368 tane non traiable lar mu ve std dir
        
"""

print([(var.name, var.trainable) for var in model.layers[1].variables]) # gradient akiyor mu bakmak icin 