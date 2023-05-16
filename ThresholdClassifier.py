
import matplotlib.pyplot as plt
list=[(0, 0),(0, 1), (1, 0), (1, 1)]
list_length=(len(list))
a=[]
b=[]

for x in list: 
  a.append(x[0])
  b.append(x[1])

plt.plot(a,b)
plt.scatter(a,b)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Four Inputs')

def and_function(x,y):
  if x==1: 
    if y==1:
      return True
    else:
      return False
  else:
    return False

def or_function(x,y):
  if x==1: 
      return True
  if y==1:
      return True
  else:
    return False

def xor_function(x,y):
  if x==1:
    if y==1:
      return False
    else:
      return True
  if y==1:
    return True
  else: 
    return False

import matplotlib.pyplot as plt
list=[(0, 0),(0, 1), (1, 0), (1, 1)]
list_length=(len(list))
a=[]
b=[]

for x in list: 
  a.append(x[0])
  b.append(x[1])

plt.plot(a,b)
plt.scatter(a,b)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Four Inputs')

first_input=int(input("give a number:"))
second_input=int(input("give a number:"))
response1=or_function(first_input,second_input)
response2=xor_function(first_input,second_input)
response3=and_function(first_input,second_input)
print('For the or function:',response1)
print('For the xor function:',response2)
print('For the and function:',response3)

or_output=[]
and_output=[]
xor_output=[]
for x in list:
  or_output.append(or_function(x[0],x[1]))
  and_output.append(and_function(x[0],x[1]))
  xor_output.append(xor_function(x[0],x[1]))

print('For the or function with the given data:',or_output)
print('For the xor function with the given data:',xor_output)
print('For the and function with the given data:',and_output)


C1=[(2, 3), (3, 3), (3, 4), (1, 4), (4, 1), (4, 3)]
C2=[(0, 0), (0, 3), (1, 1), (1, 2), (2, 1), (2, 2)]

C1_x=[]
C1_y=[]
C2_x=[]
C2_y=[]

for t in C1:
  C1_x.append(t[0])
  C1_y.append(t[1])

for t in C2:
  C2_x.append(t[0])
  C2_y.append(t[1])

plt.plot(C1_x,C1_y,'bd',label="C1")
plt.plot(C2_x, C2_y,'g^',label="c2")
plt.legend(loc="upper left")
plt.xlabel('x')
plt.ylabel('y')
plt.title('Two Data Sets')

import numpy as np
import matplotlib.pyplot as plt
import sys
def threshold_entry():
  thresh = {
          'thresh1':2,
          'thresh2':2
          }
  while True:
    thresh['thresh1']= (input("Enter the first threshold,or press x to exit: "))
    thresh['thresh2']= (input("Enter the second threshold, or press x to exit: "))
    if thresh['thresh1'].isnumeric() and thresh['thresh2'].isnumeric:
        return(thresh)
    if thresh['thresh1']=='x' or thresh['thresh2']=='x':
        return(thresh)
    else:
            print("You did not input a digit, try again")

def classification(threshold1,threshold2,C1,C2):
  lists = {
          'belongs_C1':[],
          'belongs_C2':[]
          }

  for p in C1: 
    if p[0]>=int(threshold1) and p[1]>=int(threshold2):
        lists['belongs_C1'].append([p[0],p[1]])
    else:
       lists['belongs_C2'].append([p[0],p[1]])
  for q in C2:
    if q[0]>=int(threshold1) and q[1]>=int(threshold2):
      lists['belongs_C1'].append([q[0],q[1]])
    else: 
      lists['belongs_C2'].append([q[0],q[1]])
  return lists


def accuracy(belongs_C1,belongs_C2,data_sample,C1,C2):
  correct=0
  for x in belongs_C1:
    value=x[0]
    value2=x[1]
    for t in C1:
      if value==t[0]:
        if value2==t[1]:
         correct+=1
           
  for y in belongs_C2:
    value=y[0]
    value2=y[1]
    for t in C2:
      if value==t[0]:
        if value2==t[1]:
         correct+=1
  total=(correct/data_sample)*100
  return total 

def plot(C1,C2,th1,th2):
  C1_x=[]
  C1_y=[]
  C2_x=[]
  C2_y=[]

  for t in C1:
    C1_x.append(t[0])
    C1_y.append(t[1])

  for t in C2:
    C2_x.append(t[0])
    C2_y.append(t[1])
    
  plt.plot(C1_x,C1_y,'rs',label='C1')
  plt.plot(C2_x, C2_y,'b^',label='C2')
  plt.axvline(x=int(th1),linestyle='dashed')
  plt.axhline(y=int(th2), linestyle='dashed')
  plt.xlim(-1,5)
  plt.legend(loc="upper left")
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Two Data Sets')
  plt.show()


C1=[(2, 3), (3, 3), (3, 4), (1, 4), (4, 1), (4, 3)]
C2=[(0, 0), (0, 3), (1, 1), (1, 2), (2, 1), (2, 2)]

while True:
  data_sample=len(C1)+len(C2)
  thresholds=(threshold_entry())
  if thresholds['thresh1']=='x' or thresholds['thresh2']=='x':
     sys.exit("You pressed x so the code will exit")
  lists=classification(thresholds['thresh1'],thresholds['thresh2'],C1,C2)
  total=accuracy(lists['belongs_C1'],lists['belongs_C2'], data_sample,C1,C2)
  print("For thresholds {} and {} the total accuracy is {:.2f}".format(thresholds['thresh1'],thresholds['thresh2'],total))
  plot(lists['belongs_C1'],lists['belongs_C2'],thresholds['thresh1'],thresholds['thresh2'])
