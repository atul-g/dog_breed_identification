#Download the dataset from the site: vision.stanford.edu/aditya86/ImageNetDogs/main.html

############################# IMAGE FILE PREPROCESSING ##################################

#To extract the downloaded tar file
import tarfile
tf=tarfile.open('images.tar')
tf.extractall()  #this will give the Images folder containg all the breed folders in the current directory

import os

#Renaming the 120 folders in a simpler format, like : "n02085620-Chihuahua to "Chihuahua"
for folder in os.listdir(os.path.abspath('.')+'/Images'):
    dest = folder[10:]
    dst=dest.split('_')
    dst=' '.join(dst)
    dst=os.path.abspath('.')+'/Images/'+dst
    src = os.path.abspath('.')+'/Images/'+folder
    os.rename(src, dst)  #Moves the file

#for renaming all image files in the 120 folders in a simpler format, like : "n02085620_7.jpg" to "0.jpg"
import os
cur_dir = os.path.abspath('.')
for folder in os.listdir(cur_dir+'/Images/'):
    i=0
    for imgs in os.listdir(cur_dir+'/Images/'+folder):
        src=cur_dir+'/Images/'+folder+'/'
        os.rename(src+imgs, src+str(i)+'.jpg')
        i=i+1

#Separating the dataset into training and test folders
import os
import glob
basedir = os.path.abspath('.')+'/Images/'
os.mkdir(os.path.abspath('.') + '/train')
os.mkdir(os.path.abspath('.') + '/test')
for folder in os.listdir(basedir):
    img_list = glob.glob(basedir + folder + '/*')
    ln = len(img_list)
    tr_len = int(0.8*ln) # 80% in training set and remaining 20% in test set
    tr_list = img_list[:tr_len]
    te_list = img_list[tr_len:]
    os.mkdir(os.path.abspath('.')+'/train/'+folder)
    os.mkdir(os.path.abspath('.')+'/test/'+folder)
    i,j=0,0

    for tr_img in tr_list:
        os.rename(tr_img, os.path.abspath('.')+'/train/'+folder+'/'+str(i)+'.jpg')
        i=i+1
    for te_img in te_list:
        os.rename(te_img, os.path.abspath('.')+ '/test/'+folder+'/'+str(j)+'.jpg')
        j=j+1
        


#After coding the above section, you will have two directories "train" and "test" which will
#contain subdirectories of the breeds wihch will in turn contain the images named "0.jpg, 1.jpg, 2.jpg..etc"

#Image augmentation

import os
import keras
from keras.preprocessing.image import ImageDataGenerator


image_gen_train = ImageDataGenerator(rescale = 1./255, horizontal_flip = True, zoom_range = 0.2, rotation_range=90, width_shift_range=0.2, height_shift_range=0.2)

train_gen = image_gen_train.flow_from_directory(batch_size=32, directory = os.path.abspath('.')+'/train', shuffle = True, target_size =(150, 150), class_mode = 'sparse')

image_gen_test = ImageDataGenerator(rescale = 1./255)

test_gen = image_gen_test.flow_from_directory(batch_size=32, directory=os.path.abspath('.')+'/test', target_size=(150, 150), class_mode='sparse')


################################ MODEL CREATION ####################################

#Creating the model and compiling it
from keras.applications.vgg19 import VGG19 #using VG19 model
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten


base_model = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

x = base_model.output
x = Flatten()(x)
predictions = Dense(120, activation='softmax')(x)


model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

    
model.compile(loss='sparse_categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])


model.summary()

################################## MODEL EXECUTION #####################################
#training the model
EPOCHS = 50
history = model.fit_generator(train_gen,
                    steps_per_epoch=513,         
                    epochs=EPOCHS,
                    validation_data=test_gen,
                    validation_steps=130)
