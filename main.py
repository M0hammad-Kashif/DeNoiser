import os

from tensorflow.python.ops import io_ops

from utils import *

st.title('Noise Enhancement using noise suppression')

model = tf.keras.models.load_model("Model.h5")
file_uploader = st.sidebar.file_uploader(label="", type=".wav")

st.subheader("")
st.subheader("Input Speech Sample")

def _bytes_feature(value):                                                      
  """Returns a bytes_list from a string / byte."""                              
  if isinstance(value, type(tf.constant(0))):                                   
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value])) 

if file_uploader is not None:
    # with open(file_uploader.name, "wb") as f:
    #     f.write(file_uploader.getbuffer())
    #     st.write(type(file_uploader.read()))
    #     st.audio((file_uploader.read()))
    # handle_uploaded_audio_file(file_uploader)

    out = predict(file_uploader)

    st.subheader("")
    st.subheader("")
    # st.write(type(out))

    wav_encoder = tf.audio.encode_wav(out, 16000)

    
    test = wav_encoder.numpy()

    # st.write("Test : ",type(test))

    st.subheader("Output Speech Sample")

    st.audio(test)

    # st.audio(test)

    # st.audio(wav_encoder
    # st.write(type(wav_encoder))
    # wav_saver = io_ops.write_file('output.wav', wav_encoder)

    # audio_file = open('output.wav', 'rb')
    # audio_bytes = audio_file.read()


    # st.audio(audio_bytes, format='audio/wav')
