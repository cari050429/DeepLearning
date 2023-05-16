
from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def img_plt(images, labels):
  plt.figure()
  for i in range(1,11):
    plt.subplot(2,5,i)
    plt.imshow(images[i-1,:,:],cmap='gray')
    plt.title('Label: ' + str(labels[i-1]))
  plt.show()



def feat_plot(features,labels,classes):
  for class_i in classes:
    plt.plot(features[labels[:]==classes[class_i],0],features[labels[:]==classes[class_i],1],'o',markersize=15)
  #plt.axis([-2,2,-2,2])
  plt.xlabel('x: feature1')
  plt.ylabel('y: feature2')
  plt.legend(['Class'+str(classes[class_i]) for class_i in classes])
  plt.show()

def acc_fun(labels_actual, labels_pred):
  acc=np.sum(labels_actual==labels_pred)/len(labels_actual)*100
  return acc

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
classes=np.arange(10)
print(x_train.shape)

########
#a.)Use cifar10 function in keras.datasets to load CIFAR-10 dataset. Split it into the
#training and testing sets. Define a validation set by randomly selecting 20% of the
#training images along with their corresponding labels. This will be the
#“validation_data”.

num_train_img=x_train.shape[0]
train_ind=np.arange(0,num_train_img)
train_ind_s=np.random.permutation(train_ind)
x_train=x_train[train_ind_s,:,:,:]
y_train=y_train[train_ind_s]

x_val=x_train[0:int(0.2*num_train_img),:,:,:]
y_val=y_train[0:int(0.2*num_train_img)]

x_train=x_train[int(0.2*num_train_img):,:,:]
y_train=y_train[int(0.2*num_train_img):]

print('Samples of the training images')
img_plt(x_train[0:10,:,:,:],y_train[0:10])

############
#b.)Scale the pixel values of the images in all the sets to a value between 0 and 1.
#Perform this process by dividing the image values with 255. Note: No need to flatten
#the images.

x_train=x_train.astype('float32')
x_val=x_val.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_val/=255
x_test/=255


x_train_f=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3]))
x_val_f=np.reshape(x_val,(x_val.shape[0],x_val.shape[1]*x_val.shape[2]*x_train.shape[3]))
x_test_f=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_train.shape[3]))


############
#c.)Convert the label vectors for all the sets to binary class matrices using
#to_categorical() Keras function.

y_train_c=to_categorical(y_train, len(classes))
y_val_c=to_categorical(y_val, len(classes))
y_test_c=to_categorical(y_test, len(classes))

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D, Average
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

def plot_curve(accuracy_train,loss_train,accuracy_val,loss_val):
  epochs=np.arange(loss_train.shape[0])
  plt.subplot(1,2,1)
  plt.plot(epochs,accuracy_train,epochs,accuracy_val)
  #plt.axis([-1,2,-1,2])
  plt.xlabel('Epoch#')
  plt.ylabel('Accuracy')
  plt.title('Accuracy')
  plt.legend(['Training','Validation'])

  plt.subplot(1,2,2)
  plt.plot(epochs,loss_train,epochs,loss_val)
  plt.xlabel('Epoch#')
  plt.ylabel('Binary crossentropu loss')
  plt.title('Loss')
  plt.legend(['Training','Validation'])
  plt.show()

#d.)Using Keras library, build a CNN with the following design: 2 convolutional blocks,
#1 flattening layer,1 FC layer with 512 nodes, and 1output layer. Each convolutional
#block consists of two back-to-back Conv layers followed by max pooling. The filter
#size is 3x3x image_depth. The number of filters is 32 in the first convolutional block
#and 64 in the second block.

#defining the model
model_a=Sequential()
model_a.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model_a.add(Activation('relu'))
model_a.add(Conv2D(32,(3,3)))
model_a.add(Activation('relu'))
model_a.add(MaxPooling2D(pool_size=(2,2)))

model_a.add(Conv2D(64,(3,3),padding='same'))
model_a.add(Activation('relu'))
model_a.add(Conv2D(64,(3,3)))
model_a.add(Activation('relu'))
model_a.add(MaxPooling2D(pool_size=(2,2)))

model_a.add(Flatten())
model_a.add(Dense(units=512,activation='relu'))
model_a.add(Dropout(0.5))
model_a.add(Dense(units=len(classes),activation='softmax'))
model_a.summary()

opt=tf.keras.optimizers.Adam(learning_rate=0.0001)
model_a.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

save_path='/content/drive/My Drive/model'
callback_save=ModelCheckpoint(save_path,monitor='val_loss',verbose=0,save_best_only=True,save_freq='epoch')

history=model_a.fit(x_train,y_train_c,
                    batch_size=32,
                    epochs=50,
                    verbose=1,
                    validation_data=(x_val, y_val_c),
                    callbacks=[callback_save])

plt.figure(figsize=[9,5])
acc_curve_train=np.array(history.history['accuracy'])
loss_curve_train=np.array(history.history['loss'])
acc_curve_val=np.array(history.history['val_accuracy'])
loss_curve_val=np.array(history.history['val_loss'])
plot_curve(acc_curve_train,loss_curve_train,acc_curve_val,loss_curve_val)

model_a=load_model(save_path)

#Evaluating the model on the training samples
score=model_a.evaluate(x_train,y_train_c)
print('Total loss on training set: ',score[0])
print('Accuracy of training set: ', score[1])

#Evaluating the model on the validation samples
score=model_a.evaluate(x_val,y_val_c)
print('Total loss on testing set:', score[0])
print('Accuracy of testing set',score[1])

from keras.utils.vis_utils import plot_model,model_to_dot
plot_model(model_a,to_file='model_cnn.png')

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D, Average
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from keras.preprocessing.image import ImageDataGenerator

def plot_curve(accuracy_train,loss_train,accuracy_val,loss_val):
  epochs=np.arange(loss_train.shape[0])
  plt.subplot(1,2,1)
  plt.plot(epochs,accuracy_train,epochs,accuracy_val)
  #plt.axis([-1,2,-1,2])
  plt.xlabel('Epoch#')
  plt.ylabel('Accuracy')
  plt.title('Accuracy')
  plt.legend(['Training','Validation'])

  plt.subplot(1,2,2)
  plt.plot(epochs,loss_train,epochs,loss_val)
  plt.xlabel('Epoch#')
  plt.ylabel('Binary crossentropu loss')
  plt.title('Loss')
  plt.legend(['Training','Validation'])
  plt.show()

#d.)Using Keras library, build a CNN with the following design: 2 convolutional blocks,
#1 flattening layer,1 FC layer with 512 nodes, and 1output layer. Each convolutional
#block consists of two back-to-back Conv layers followed by max pooling. The filter
#size is 3x3x image_depth. The number of filters is 32 in the first convolutional block
#and 64 in the second block.

#defining the model
model_a=Sequential()
model_a.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model_a.add(Activation('relu'))
model_a.add(Conv2D(32,(3,3)))
model_a.add(Activation('relu'))
model_a.add(MaxPooling2D(pool_size=(2,2)))

model_a.add(Conv2D(32,(3,3),padding='same'))
model_a.add(Activation('relu'))
model_a.add(Conv2D(64,(3,3)))
model_a.add(Activation('relu'))
model_a.add(MaxPooling2D(pool_size=(2,2)))

model_a.add(Flatten())
model_a.add(Dense(units=512,activation='relu'))
model_a.add(Dropout(0.5))
model_a.add(Dense(units=len(classes),activation='softmax'))
model_a.summary()

opt=tf.keras.optimizers.Adam(learning_rate=0.0001)
model_a.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

datagen= ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

datagen.fit(x_train)

save_path='/content/drive/My Drive/model'
callback_save=ModelCheckpoint(save_path,monitor='val_loss',verbose=0,save_best_only=True,save_freq='epoch')

history=model_a.fit(datagen.flow(x_train,y_train_c, batch_size=32),
                                 steps_per_epoch=len(x_train)/32, epochs=50, 
                                 verbose=1, validation_data=(x_val,y_val_c),callbacks=[callback_save])

plt.figure(figsize=[9,5])
acc_curve_train=np.array(history.history['accuracy'])
loss_curve_train=np.array(history.history['loss'])
acc_curve_val=np.array(history.history['val_accuracy'])
loss_curve_val=np.array(history.history['val_loss'])
plot_curve(acc_curve_train,loss_curve_train,acc_curve_val,loss_curve_val)

model_a=load_model(save_path)

#Evaluating the model on the training samples
score=model_a.evaluate(x_train,y_train_c)
print('Total loss on training set: ',score[0])
print('Accuracy of training set: ', score[1])

#Evaluating the model on the validation samples
score=model_a.evaluate(x_val,y_val_c)
print('Total loss on testing set:', score[0])
print('Accuracy of testing set',score[1])

"""g.)In the e, the model starts overfitting at 5-10 epoch which can be seen from the validation loss. However, the data augmentation that was incorporated into f shows how this overfitting gets better with augmentation. """

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D, Average, BatchNormalization
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score

def plot_curve(accuracy_train,loss_train,accuracy_val,loss_val):
  epochs=np.arange(loss_train.shape[0])
  plt.subplot(1,2,1)
  plt.plot(epochs,accuracy_train,epochs,accuracy_val)
  #plt.axis([-1,2,-1,2])
  plt.xlabel('Epoch#')
  plt.ylabel('Accuracy')
  plt.title('Accuracy')
  plt.legend(['Training','Validation'])

  plt.subplot(1,2,2)
  plt.plot(epochs,loss_train,epochs,loss_val)
  plt.xlabel('Epoch#')
  plt.ylabel('Loss')
  plt.title('Loss')
  plt.legend(['Training','Validation'])
  plt.show()

#d.)Using Keras library, build a CNN with the following design: 2 convolutional blocks,
#1 flattening layer,1 FC layer with 512 nodes, and 1output layer. Each convolutional
#block consists of two back-to-back Conv layers followed by max pooling. The filter
#size is 3x3x image_depth. The number of filters is 32 in the first convolutional block
#and 64 in the second block.

#defining the model
model_a=Sequential()
model_a.add(Conv2D(32,(3,3),padding='same',input_shape=x_train.shape[1:]))
model_a.add(BatchNormalization())
model_a.add(Activation('relu'))
model_a.add(Conv2D(32,(3,3)))
model_a.add(BatchNormalization())
model_a.add(Activation('relu'))
model_a.add(MaxPooling2D(pool_size=(2,2)))

model_a.add(Conv2D(32,(3,3),padding='same'))
model_a.add(BatchNormalization())
model_a.add(Activation('relu'))
model_a.add(Conv2D(64,(3,3)))
model_a.add(BatchNormalization())
model_a.add(Activation('relu'))
model_a.add(MaxPooling2D(pool_size=(2,2)))

model_a.add(Flatten())
model_a.add(Dense(units=512))
model_a.add(BatchNormalization())
model_a.add(Activation('relu'))
model_a.add(Dropout(0.5))
model_a.add(Dense(units=len(classes)))
model_a.add(BatchNormalization())
model_a.add(Activation('softmax'))
model_a.summary()

opt=tf.keras.optimizers.Adam(learning_rate=0.01)
model_a.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

save_path='/content/drive/My Drive/model'
callback_save=ModelCheckpoint(save_path,monitor='val_loss',verbose=0,save_best_only=True,save_freq='epoch')

history=model_a.fit(x_train,y_train_c,
                    batch_size=64,
                    epochs=50,
                    verbose=1,
                    validation_data=(x_val, y_val_c),
                    callbacks=[callback_save])

plt.figure(figsize=[9,5])
acc_curve_train=np.array(history.history['accuracy'])
loss_curve_train=np.array(history.history['loss'])
acc_curve_val=np.array(history.history['val_accuracy'])
loss_curve_val=np.array(history.history['val_loss'])
plot_curve(acc_curve_train,loss_curve_train,acc_curve_val,loss_curve_val)

model_a=load_model(save_path)

#Evaluating the model on the training samples
score=model_a.evaluate(x_train,y_train_c)
print('Total loss on training set: ',score[0])
print('Accuracy of training set: ', score[1])

#Evaluating the model on the validation samples
score=model_a.evaluate(x_val,y_val_c)
print('Total loss on testing set:', score[0])
print('Accuracy of testing set',score[1])

"""i.)The training loss in h decreases much quicker than the training loss in e. This could be because of the normalization layer which allows you to use higher learning rates. Higher learning rates in only 50 epochs means that the model can train in less time, just not as accurately. """
