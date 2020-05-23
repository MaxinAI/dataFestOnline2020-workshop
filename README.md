# COVID 19: Detect Face Mask in Real Time

Repo contains slides and code for workshop of DataFestOnline 2020. 

The slides present convolutional neural networks and code used for model training to detect face mask on person. 

# Authors:

Soso Sukhitashvili - [linkedin](https://www.linkedin.com/in/soso-sukhitashvili/)

Nodo Okroshiashvili - [linkedin](https://www.linkedin.com/in/nodar-okroshiashvili/)

## Getting Started

Git clone this repo

### Installing

Create conda env and install requirements:

``
conda create --name workshop python==3.7
``

``
conda activate workshop
``

``
pip install -r requirements.txt
``
### Examples

Run mask detection code: `` python test_video.py``

# References

- [Tutorial](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148) about convolutional neural networks

- [Tutorial](https://towardsdatascience.com/how-i-built-a-face-mask-detector-for-covid-19-using-pytorch-lightning-67eb3752fd61) to built a face mask detector for COVID-19 using [PyTorch](https://pytorch.org/)

- [Tutorial](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/) to built a face mask detector for COVID-19 using [TensorFlow](https://www.tensorflow.org/)


# Credits

Face mask detection and model training code is taken from this nice [tutorial](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/).