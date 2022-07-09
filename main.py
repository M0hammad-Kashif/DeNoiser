import os
import tensorflow as tf
from tensorflow.python.ops import io_ops
# from IPython.display import Audio
# from IPython.core.display import display

from utils import *

st.title('Noise Enhancement using noise suppression')

# audio_file = open('sampleAudio/sample.m4a', 'rb')
# audio_bytes = audio_file.read()
#
# st.audio(audio_bytes, format='audio/m4a')

# Taking a file (audio) for uploading
# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     file_details = {
#         "Filename": uploaded_file.name,
#         "Filetype": uploaded_file.type
#     }
#
#     st.write(file_details)  # optional
#     _, input_audio = audio_to_display(uploaded_file)
#     st.audio(input_audio, format='audio/m4a')
model = tf.keras.models.load_model("Model.h5")
file_uploader = st.sidebar.file_uploader(label="", type=".wav")

st.subheader("")
st.subheader("Input Speech Sample")

if file_uploader is not None:
    # st.write(file_uploader)
    with open(os.path.join("./savedFiles", file_uploader.name), "wb") as f:
        f.write(file_uploader.getbuffer())
    handle_uploaded_audio_file(file_uploader)
    st.subheader("")
    st.subheader("")


    # st.write(get_audio(os.path.join("./savedFiles", file_uploader.name)))

    out = predict(os.path.join("./savedFiles", file_uploader.name))

    wav_encoder = tf.audio.encode_wav(out, 16000)
    # st.write()
    wav_saver = io_ops.write_file('./outputFiles/output.wav', wav_encoder)

    
    audio_file = open('./outputFiles/output.wav', 'rb')
    audio_bytes = audio_file.read()

    st.subheader("Output Speech Sample")
    
    st.audio(audio_bytes, format='audio/wav')
    
    # st.write(out)

# operation on model
# interpreter = tf.lite.Interpreter(model_path='TFLiteModel.tflite')
# interpreter.allocate_tensors()
#
# predict_tflite()
