import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from tensorflow.keras.layers import Layer, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Multiply, Conv2D, Activation, Add, Concatenate, LayerNormalization, MultiHeadAttention
from tensorflow.keras import backend as K
from flask import Flask, request, jsonify
import os
from PIL import Image
import io

# ------------------------------
# CBAM Block
# ------------------------------
@tf.keras.utils.register_keras_serializable()
class CBAMBlock(Layer):
    def __init__(self, filters, reduction_ratio=16, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.filters = filters
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        self.shared_dense_one = Dense(self.filters // self.reduction_ratio,
                                      activation='relu',
                                      kernel_initializer='he_normal',
                                      use_bias=True)
        self.shared_dense_two = Dense(self.filters,
                                      kernel_initializer='he_normal',
                                      use_bias=True)
        self.conv_spatial = Conv2D(filters=1,
                                   kernel_size=7,
                                   strides=1,
                                   padding='same',
                                   activation='sigmoid',
                                   kernel_initializer='he_normal',
                                   use_bias=False)

    def call(self, input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        max_pool = GlobalMaxPooling2D()(input_tensor)

        avg_pool = self.shared_dense_one(avg_pool)
        avg_pool = self.shared_dense_two(avg_pool)

        max_pool = self.shared_dense_one(max_pool)
        max_pool = self.shared_dense_two(max_pool)

        cbam_channel = Add()([avg_pool, max_pool])
        cbam_channel = Activation('sigmoid')(cbam_channel)
        cbam_channel = Reshape((1, 1, self.filters))(cbam_channel)
        channel_refined = Multiply()([input_tensor, cbam_channel])

        avg_pool_spatial = K.mean(channel_refined, axis=-1, keepdims=True)
        max_pool_spatial = K.max(channel_refined, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg_pool_spatial, max_pool_spatial])
        cbam_spatial = self.conv_spatial(concat)

        refined_feature = Multiply()([channel_refined, cbam_spatial])
        return refined_feature

    def get_config(self):
        config = super(CBAMBlock, self).get_config()
        config.update({
            "filters": self.filters,
            "reduction_ratio": self.reduction_ratio
        })
        return config

# ------------------------------
# Tiny Transformer Block
# ------------------------------
@tf.keras.utils.register_keras_serializable()
class TinyTransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TinyTransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TinyTransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.att.key_dim,
            "num_heads": self.att.num_heads,
            "ff_dim": self.ffn[0].units
        })
        return config

# ------------------------------
# Load Model with custom_objects
# ------------------------------
model = load_model(
    "model.h5",
    custom_objects={'CBAMBlock': CBAMBlock, 'TinyTransformerBlock': TinyTransformerBlock}
)

class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']

# ------------------------------
# Grad-CAM Function
# ------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ------------------------------
# Create Flask app
# ------------------------------
app = Flask(__name__)

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def home():
    return "Brain Tumor Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."})

    file = request.files['file']

    img = Image.open(file.stream).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    label = class_labels[predicted_class]

    result_text = f"Tumor predicted: {label}" if label != "no tumor" else "No tumor predicted"
    
    return jsonify({"prediction": result_text})

# ------------------------------
# Heroku-Compatible Launch
# ------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
