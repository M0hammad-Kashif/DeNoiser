import streamlit as st
import tensorflow as tf
from utils import *

st.title('Noise Enhancement using noise suppression')

audio_file = open('sampleAudio/sample.m4a', 'rb')
audio_bytes = audio_file.read()

st.audio(audio_bytes, format='audio/m4a')

# operation on model
interpreter = tf.lite.Interpreter(model_path='TFLiteModel.tflite')
interpreter.allocate_tensors()

predict_tflite()
