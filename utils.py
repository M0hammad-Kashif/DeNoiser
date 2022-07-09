import tensorflow as tf

from main import interpreter, model

batching_size = 12000


def get_audio(path):
    audio, _ = tf.audio.decode_wav(tf.io.read_file(path), 1)
    return audio


def inference_preprocess(path):
    audio = get_audio(path)
    audio_len = audio.shape[0]
    batches = []
    for i in range(0, audio_len - batching_size, batching_size):
        batches.append(audio[i:i + batching_size])

    batches.append(audio[-batching_size:])
    diff = audio_len - (i + batching_size)
    return tf.stack(batches), diff


def predict(path):
    test_data, diff = inference_preprocess(path)
    predictions = model.predict(test_data)
    final_op = tf.reshape(predictions[:-1], ((predictions.shape[0] - 1) * predictions.shape[1], 1))
    final_op = tf.concat((final_op, predictions[-1][-diff:]), axis=0)
    return final_op


def predict_tflite(path):
    test_audio, diff = inference_preprocess(path)
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    preds = []
    for i in test_audio:
        interpreter.set_tensor(input_index, tf.expand_dims(i, 0))
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_index)
        preds.append(predictions)

    predictions = tf.squeeze(tf.stack(preds, axis=1))
    final_op = tf.reshape(predictions[:-1], ((predictions.shape[0] - 1) * predictions.shape[1], 1))
    final_op = tf.concat((tf.squeeze(final_op), predictions[-1][-diff:]), axis=0)
    return final_op
