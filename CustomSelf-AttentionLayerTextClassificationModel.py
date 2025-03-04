import tensorflow as tf
from tensorflow.keras import layers, Model

# Define a custom self-attention layer
class SelfAttention(layers.Layer):
    def __init__(self, units):
        super(SelfAttention, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        self.W_query = self.add_weight(shape=(input_shape[-1], self.units),
                                       initializer='glorot_uniform',
                                       trainable=True)
        self.W_key = self.add_weight(shape=(input_shape[-1], self.units),
                                     initializer='glorot_uniform',
                                     trainable=True)
        self.W_value = self.add_weight(shape=(input_shape[-1], self.units),
                                       initializer='glorot_uniform',
                                       trainable=True)
        super(SelfAttention, self).build(input_shape)
    
    def call(self, inputs):
        # inputs shape: (batch_size, seq_len, input_dim)
        query = tf.tensordot(inputs, self.W_query, axes=[[2], [0]])  # (batch, seq_len, units)
        key   = tf.tensordot(inputs, self.W_key, axes=[[2], [0]])
        value = tf.tensordot(inputs, self.W_value, axes=[[2], [0]])
        
        # Compute attention scores
        scores = tf.matmul(query, key, transpose_b=True)  # (batch, seq_len, seq_len)
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = scores / tf.math.sqrt(d_k)
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)
        output = tf.matmul(attention_weights, value)  # (batch, seq_len, units)
        return output

# Example text classification model using the custom self-attention layer
vocab_size = 5000
embedding_dim = 128
max_length = 100
num_classes = 2

inputs = layers.Input(shape=(max_length,))
embedding = layers.Embedding(vocab_size, embedding_dim, input_length=max_length)(inputs)
# Apply custom self-attention
attention_out = SelfAttention(units=64)(embedding)
pooled = layers.GlobalAveragePooling1D()(attention_out)
dense = layers.Dense(64, activation='relu')(pooled)
outputs = layers.Dense(num_classes, activation='softmax')(dense)

attention_model = Model(inputs, outputs)
attention_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
attention_model.summary()
