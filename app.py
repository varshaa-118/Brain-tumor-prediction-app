import gradio as gr
import numpy as np
import tensorflow as tf
import psycopg2
import os
import gdown
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---- Download model from Google Drive if not already present ----

model_path = "model/brain_tumor_model_with_gradcam.h5"
gdrive_url = "https://drive.google.com/uc?id=19JgIusns4ZBoJQAHDUYUXj2Ujfh0bBuM"

if not os.path.exists(model_path):
    os.makedirs("model", exist_ok=True)
    print("Downloading model from Google Drive...")
    gdown.download(gdrive_url, model_path, quiet=False)

# ---- Custom Layers ----

class TinyTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()

    def call(self, inputs):
        attn_output = self.attn(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class CBAMBlock(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPooling2D()
        self.shared_mlp = tf.keras.Sequential([
            layers.Dense(channel // self.reduction_ratio, activation='relu'),
            layers.Dense(channel)
        ])
        self.conv2d = layers.Conv2D(filters=1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = self.avg_pool(inputs)
        max_pool = self.max_pool(inputs)
        avg_out = self.shared_mlp(avg_pool)
        max_out = self.shared_mlp(max_pool)
        channel_attention = tf.nn.sigmoid(avg_out + max_out)
        channel_attention = tf.expand_dims(tf.expand_dims(channel_attention, 1), 1)
        x = inputs * channel_attention
        avg_pool_spatial = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_attention = self.conv2d(concat)
        return x * spatial_attention

# ---- Load model ----

model = load_model(model_path, custom_objects={
    'CBAMBlock': CBAMBlock,
    'TinyTransformerBlock': TinyTransformerBlock
})

last_conv_layer_name = "Conv_1"
class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ---- Database Logging ----

def insert_prediction(image_name, predicted_class, confidence):
    try:
        print(f"Inserting prediction for image: {image_name} - {predicted_class} with confidence: {confidence:.2f}")
        conn = psycopg2.connect(
            dbname="brain_tumor_log",
            user="postgres",
            password="123456",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (image_name, predicted_class, confidence_score, timestamp) VALUES (%s, %s, %s, %s)",
            (image_name, predicted_class, confidence, datetime.now())
        )
        conn.commit()
        cur.close()
        conn.close()
        print("Prediction logged to database successfully.")
    except Exception as e:
        print("Database logging error:", e)

# ---- Grad-CAM ----

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model({"input_layer": img_array})
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(image, heatmap, alpha=0.3):
    img = np.array(image.resize((224, 224)))
    heatmap = np.uint8(255 * heatmap)
    colormap = cm.get_cmap("jet")
    jet_colors = colormap(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.image.resize(jet_heatmap, (img.shape[0], img.shape[1])).numpy()
    superimposed_img = jet_heatmap * alpha + img / 255.0
    return Image.fromarray(np.uint8(superimposed_img * 255))

# ---- Predict ----

def predict(image):
    image_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0)

    predictions = model.predict({"input_layer": img_array})[0]
    pred_index = np.argmax(predictions)
    pred_class = class_names[pred_index]
    confidence = float(predictions[pred_index]) * 100

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    gradcam_img = overlay_gradcam(image_resized, heatmap)

    image_name = getattr(image, "filename", "Unknown Image")
    insert_prediction(image_name, pred_class, round(confidence, 2))

    if pred_class == "No Tumor":
        result_text = f"🧠 No tumor predicted\nConfidence: {confidence:.2f}%"
    else:
        result_text = f"⚠️ Tumor predicted: {pred_class}\nConfidence: {confidence:.2f}%"

    return result_text, gradcam_img

# ---- Gradio Interface ----

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=["text", "image"],
    title="Brain Tumor Detection with Grad-CAM",
    description="Upload a brain MRI to classify tumor type and view Grad-CAM. Logs predictions to PostgreSQL."
)

interface.launch()
