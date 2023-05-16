
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

def p_subplot(images,digit):
    plt.imshow(images, cmap='gray')
    plt.title('Label:'+str(digit))

def subplot (images,labels):
  for digit in range(10):
    i=labels[digit]
    x_train_d=images[labels==digit,:,:]
    x_train_i=x_train_d[digit,:,:]
    plt.subplot(2,5,digit+1)
    p_subplot(x_train_i,digit)
  plt.show()
    


(x_train, y_train), (x_test,y_test)=mnist.load_data()
print('x_train:'+ str(x_train.shape))
print('y_train:'+ str(y_train.shape))
print('x_test:'+ str(x_test.shape))
print('y_test:'+ str(y_test.shape))
subplot(x_train,y_train)

x_train_01=x_train[np.logical_or(y_train == 0, y_train == 8),:,:]
y_train_01=y_train[np.logical_or(y_train == 0, y_train == 8)]
x_test_01=x_test[np.logical_or(y_test == 0 , y_test == 8),:,:]
y_test_01=y_test[np.logical_or(y_test == 0, y_test == 8)]


x_valid_01=x_train_01[0:500,:,:]
y_valid_01=y_train_01[0:500]
x_train_01=x_train_01[500:,:,:]
y_train_01=y_train_01[500:]
print('x_train:'+ str(x_train_01.shape))
print('y_train:'+ str(y_train_01.shape))
print('x_valid:'+ str(x_valid_01.shape))
print('y_valid:'+ str(y_valid_01.shape))
for digit in range(10):
  plt.subplot(2,5,digit+1)
  p_subplot(x_valid_01[digit],digit)
plt.show()

attribute1=np.sum(x_train_01[:,12:16,12:16], axis=2)
attribute2=np.sum(x_valid_01[:,12:16,12:16], axis=2)
attribute3=np.sum(x_test_01[:,12:16,12:16], axis=2)

attribute1_avg=np.sum(attribute1,axis=1)/16
attribute2_avg=np.sum(attribute2,axis=1)/16
attribute3_avg=np.sum(attribute3,axis=1)/16

array=np.array(range(1,501))
array_0=array[y_valid_01==0]
array_1=array[y_valid_01==8]

plt.scatter(array_0,attribute2_avg[y_valid_01==0],c='red', label='0')
plt.scatter(array_1,attribute2_avg[y_valid_01==8],c='green', label='8')
plt.xlabel('Sample #')
plt.ylabel('Average of the 4x4 center grid')
plt.legend(loc="upper left")
plt.show()

threshold=int(input('Input a threshold:'))


correctt=0
for x in range(len(y_train_01)):
  if (attribute1_avg[x]<threshold).all() and y_train_01[x]==0:
    correctt+=1
  else:
    if (attribute1_avg[x]>=threshold).all() and y_train_01[x]==8:
      correctt+=1
taccuracy=correctt/len(y_train_01)
print("Training accuracy is:",+taccuracy)

correctv=0
for x in range(len(y_valid_01)):
  if (attribute2_avg[x]<threshold).all() and y_valid_01[x]==0:
    correctv+=1
  else:
    if (attribute2_avg[x]>=threshold).all() and y_valid_01[x]==8:
      correctv+=1
vaccuracy=correctv/len(y_valid_01)
print("Validation accuracy is:",+vaccuracy)

correctr=0
for x in range(len(y_test_01)):
  if (attribute3_avg[x]<threshold).all() and y_test_01[x]==0:
    correctr+=1
  else:
    if (attribute3_avg[x]>=threshold).all() and y_test_01[x]==8:
      correctr+=1
test_accuracy=correctr/len(y_test_01)
print("Testing accuracy is:",+test_accuracy)
