import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, datasets

# Load CIFAR-10 dataset and preprocess
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
y_train = tf.squeeze(tf.one_hot(y_train, 10))
y_test  = tf.squeeze(tf.one_hot(y_test, 10))

# Define a CNN model using subclassing
class CustomCNN(models.Model):
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), padding="same", activation="relu")
        self.bn1   = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(64, (3, 3), padding="same", activation="relu")
        self.bn2   = layers.BatchNormalization()
        self.pool  = layers.MaxPooling2D((2, 2))
        self.flatten = layers.Flatten()
        self.dense1  = layers.Dense(128, activation="relu")
        self.dropout = layers.Dropout(0.5)
        self.out     = layers.Dense(num_classes, activation="softmax")
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.out(x)

model = CustomCNN()

# Define optimizer with an exponential learning rate decay schedule
initial_lr = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr,
    decay_steps=10000,
    decay_rate=0.9,
    staircase=True
)
optimizer = optimizers.Adam(learning_rate=lr_schedule)
loss_fn = losses.CategoricalCrossentropy()

# Prepare the datasets
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(5000).batch(batch_size)
test_dataset  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Custom training loop
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    train_loss = tf.metrics.Mean()
    train_acc  = tf.metrics.CategoricalAccuracy()
    
    # Training loop
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            logits = model(x_batch, training=True)
            loss_value = loss_fn(y_batch, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        train_loss(loss_value)
        train_acc(y_batch, logits)
    
    print("Training loss:", train_loss.result().numpy(), "Accuracy:", train_acc.result().numpy())
    
    # Evaluate on test dataset
    test_acc = tf.metrics.CategoricalAccuracy()
    for x_batch, y_batch in test_dataset:
        logits = model(x_batch, training=False)
        test_acc(y_batch, logits)
    print("Test accuracy:", test_acc.result().numpy())
