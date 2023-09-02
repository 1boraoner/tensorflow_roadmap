import tensorflow as tf

def create_functional_model(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

class CustomModelWithTraining(tf.keras.Model):
    def __init__(self, functional_model, num_classes):
        super(CustomModelWithTraining, self).__init__()
        self.functional_model = functional_model
        self.num_classes = num_classes

    def call(self, x):
        print(x.shape)
        return self.functional_model(x)

    def train_step(self, inputs, labels, loss_fn, optimizer):
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = loss_fn(labels, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

# Instantiate the functional model
num_classes = 10
input_shape = (64, 64, 3)
functional_model = create_functional_model(input_shape, num_classes)

# Instantiate the custom model with training
custom_model_with_training = CustomModelWithTraining(functional_model, num_classes)

# Generate example data
train_dataset = tf.data.Dataset.from_tensor_slices(
    (tf.random.normal((100, 64, 64, 3)), tf.one_hot(tf.random.uniform((100,), maxval=num_classes, dtype=tf.int32), depth=num_classes))
)
train_dataset = train_dataset.batch(32)

# Define loss function and optimizer
loss_fn = tf.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.optimizers.Adam()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0.0
    for inputs, labels in train_dataset:
        batch_loss = custom_model_with_training.train_step(inputs, labels, loss_fn, optimizer)
        total_loss += batch_loss
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
