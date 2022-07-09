# DeNoisingAudio
<b>Problem </b> Speech denoising is a long-standing problem. We can imagine someone talking in a video conference while there is a random background noise. In this situation, a speech denoising system has the job of removing the background noise in order to improve the speech signal and there are many more situations where noise is a disturbing factor to the speech signal. This application is especially important for video and audio conferences, where noise can significantly decrease speech intelligibility.

<b>Shortcomings in existing methods</b> : The currrent noise suppression techniques uses Multi-mic system.

- Two and more mics also make the audio path and acoustic design quite difficult and expensive for device OEMs and ODMs. Audio/Hardware/Software engineers have to implement suboptimal tradeoffs to support both the industrial design and voice quality requirements

<b>Solution</b> We tackle this problem using Convolutional & Transpose Convolutional Neural Network . Given a noisy input signal from a single microphone, we aim to build a model that can extract the clean signal from the given noisy input signal and return it to the user. 