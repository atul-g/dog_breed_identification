# Dog Breed Identification using Transfer Learning

### A small project on training a neural network on the _Stanford Dogs Dataset_ using transfer learning.

Dataset can be downloaded from [here](vision.stanford.edu/aditya86/ImageNetDogs/main.html).

The dataset comprises of 120 classes/breeds of dogs with a total of 20,580 images.

I used the pre-trained [VG19](https://keras.io/applications/#vgg19) model fom keras.applications and added a dense layer to it to train the model. [ImageDataGenerator](https://keras.io/preprocessing/image/) from Keras's image preprocessing was used for image augmentation. Model was compiled and trainde for 50 epochs with a batch size of 32 (alterable).


