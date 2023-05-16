

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    
    def __init__(self,x):
        np.random.seed(1)
        self.weight_matrix=2 * np.random.random((3,1)) - 1
        print(self.weight_matrix)
        self.l_rate=x
        print(self.l_rate)
    
    def sigmoid(self,x):
      return (1/(1+np.exp(-x)))
    
    def forward_propagation(self,inputs):
        outs=np.dot(inputs,self.weight_matrix)
        return self.sigmoid(outs)
    
    def pred(self,inputs):
        prob=self.forward_propagation(inputs)
        preds=np.int8(prob>=0.5)
        return preds
    
    def train_GDL(self,train_inputs,train_outputs,num_train_iterations=1000,lr=0.1):
        N=train_inputs.shape[0]
        self.l_rate=lr
        cost_func=np.array([])
        for iteration in range(num_train_iterations):
          outputs=self.forward_propagation(train_inputs)
          error=train_outputs-outputs
          adjustment=(self.l_rate/N)*np.sum(np.dot(error,train_inputs), axis=0)
          cost_func=np.append(cost_func,(1/(2*N))*np.sum(np.power(error,2)))
          self.weight_matrix[:,0]+=adjustment
          print('Iteration #'+str(iteration))
          plot_fun_thr(train_inputs[:,1:3], train_outputs[:],self.weight_matrix[:,0],classes)
        plot_cost_fun(cost_func,num_train_iterations)

def plot_fun_thr(features,labels,thre_params,classes):
      plt.plot(features[labels[:]==classes[0],0],features[labels[:]==classes[0],1],'rs'
               ,features[labels[:]==classes[1],0],features[labels[:]==classes[1],1],'g^')
      plt.axis([-1,2,-1,2])
      x1=np.linspace(-1,2,50)
      x2=-(thre_params[1]*x1+thre_params[0])/thre_params[2]
      plt.plot(x1,x2,'-r')
      plt.xlabel('x:feature 1')
      plt.ylabel('y:feature 2')
      plt.legend(['Class'+str(classes[0]), 'Class'+str(classes[1])])
      plt.show()
      
def plot_cost_fun(J,iterations):
    x=np.arange(iterations,dtype=int)
    y=J
    plt.plot(x,y)
    plt.axis([-1,x.shape[0]+1,-1,np.max(y)+1])
    plt.title('learning curve')
    plt.xlabel('x:   iteration number')
    plt.ylabel('y:   J(0)')
    plt.show()

def plot_fun(features,labels,classes):
    plt.plot(features[labels[:]==classes[0],0],features[labels[:]==classes[0],1],'rs'
             ,features[labels[:]==classes[1],0],features[labels[:]==classes[1],1],'g^')
    plt.axis([-1,2,-1,2])
    plt.xlabel('x:feature 1')
    plt.ylabel('y:feature 2')
    plt.legend(['Class'+str(classes[0]), 'Class'+str(classes[1])])
    plt.show()

###################################################################
features =np.array([[1,1],[1,0],[0,1],[0.5,-1],[0.5,3],[0.7,2],[-1,0],[-1,1],[2,0],[0,0]])
labels=np.array([1,1,0,0,1,1,0,0,1,0])
classes=[0,1]

plot_fun(features,labels,classes)

bias=np.ones((features.shape[0],1))
print(bias)
print(bias.shape)
features=np.append(bias,features,axis=1)
print('Features vector after adding the bias')
print(features)
print(features.shape)        
    
neural_network=NeuralNetwork(1)
print('Random weights at the start of training')
print(neural_network.weight_matrix)
num_iterations=10
neural_network.train_GDL(features,labels,50,1)

print('New weights after training')
print(neural_network.weight_matrix)

print('Testing network on training data points -->')
print(neural_network.pred(features))

# classifiy the given data samples
print('Testing network on new examples-->')
#print(neural_network.pred(np.array([[1,1],[1,0],[0,1],[0.5,-1],[0.5,3],[0.7,2],[-1,0],[-1,1],[2,0],[0,0]])))
#######################################################################################################################
features_5 =np.array([[1,1],[1,0],[0,1],[0.5,-1],[0.5,3],[0.7,2],[-1,0],[-1,1],[2,0],[0,0]])
labels=np.array([1,1,0,0,1,1,0,0,1,0])
classes=[0,1]

plot_fun(features_5,labels,classes)

bias=np.ones((features_5.shape[0],1))
print(bias)
print(bias.shape)
features_5=np.append(bias,features_5,axis=1)
print('Features vector after adding the bias')
print(features_5)
print(features_5.shape)        
    
neural_network2=NeuralNetwork(0.5)
print('Random weights at the start of training')
print(neural_network2.weight_matrix)
num_iterations=10
neural_network2.train_GDL(features_5,labels,50,0.5)

print('New weights after training')
print(neural_network2.weight_matrix)

print('Testing network on training data points -->')
print(neural_network2.pred(features))
##############################################################
features_1 =np.array([[1,1],[1,0],[0,1],[0.5,-1],[0.5,3],[0.7,2],[-1,0],[-1,1],[2,0],[0,0]])
labels=np.array([1,1,0,0,1,1,0,0,1,0])
classes=[0,1]

plot_fun(features_1,labels,classes)

bias=np.ones((features_1.shape[0],1))
print(bias)
print(bias.shape)
features_1=np.append(bias,features_1,axis=1)
print('Features vector after adding the bias')
print(features_1)
print(features_1.shape)        
    
neural_network3=NeuralNetwork(0.1)
print('Random weights at the start of training')
print(neural_network3.weight_matrix)
num_iterations=10
neural_network3.train_GDL(features_1,labels,50,0.1)

print('New weights after training')
print(neural_network3.weight_matrix)

print('Testing network on training data points -->')
print(neural_network3.pred(features))
###############################################################
features_01 =np.array([[1,1],[1,0],[0,1],[0.5,-1],[0.5,3],[0.7,2],[-1,0],[-1,1],[2,0],[0,0]])
labels=np.array([1,1,0,0,1,1,0,0,1,0])
classes=[0,1]

plot_fun(features_01,labels,classes)

bias=np.ones((features_01.shape[0],1))
print(bias)
print(bias.shape)
features_01=np.append(bias,features_01,axis=1)
print('Features vector after adding the bias')
print(features_01)
print(features_01.shape)        
    
neural_network4=NeuralNetwork(0.01)
print('Random weights at the start of training')
print(neural_network4.weight_matrix)
num_iterations=10
neural_network4.train_GDL(features_01,labels,50,0.01)

print('New weights after training')
print(neural_network4.weight_matrix)

print('Testing network on training data points -->')
print(neural_network4.pred(features))
