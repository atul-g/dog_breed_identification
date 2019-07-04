# Dog Breed Identification using Transfer Learning

A small project on training a neural network on the 'Stanford Dogs Dataset' using transfer learning.

Dataset was downloaded from vision.stanford.edu/aditya86/ImageNetDogs/main.html
The dataset comprises of 120 classes/breeds of dogs with a total of 20,580 images.

I used the pre-trained VG19 model fom keras.applications and added a dense layer to it to train the model. ImageDataGenerator from Keras's image preprocessing was used for image augmentation. Model was compiled for 50 epochs with a batch size of 32 (alterable).


