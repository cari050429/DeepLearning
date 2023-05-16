

import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(object):
    
    def __init__(self):
        np.random.seed(1)
        self.weight_matrix=2 * np.random.random((3,1)) - 1
        self.l_rate=1
    
    def hard_limiter(self,x):
        outs=np.zeros(x.shape)
        outs[x>0]=1
        return outs
    
    def forward_progration(self,inputs):
        outs=np.dot(inputs,self.weight_matrix)
        return self.hard_limiter(outs)
    
    def pred(self,inputs):
        preds=self.forward_progration(inputs)
        return preds
    
    def train(self,train_inputs,train_outputs,num_train_iterations=1000):
        for iteration in range(num_train_iterations):
            for i in range(train_inputs.shape[0]):
                pred_i=self.pred(train_inputs[i,:])
                if pred_i!=train_outputs[i]:
                    output=self.forward_progration(train_inputs[i,:])
                    error=train_outputs[i]-output
                    adjustment=self.l_rate*error*train_inputs[i]
                    self.weight_matrix[:,0]+=adjustment
            print('Iteration #'+str(iteration))
            plot_fun_thr(train_inputs[:,1:3],train_outputs,self.weight_matrix[:,0],classes)

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
      
def plot_fun(features,labels,classes):
    plt.plot(features[labels[:]==classes[0],0],features[labels[:]==classes[0],1],'rs'
             ,features[labels[:]==classes[1],0],features[labels[:]==classes[1],1],'g^')
    plt.axis([-1,2,-1,2])
    plt.xlabel('x:feature 1')
    plt.ylabel('y:feature 2')
    plt.legend(['Class'+str(classes[0]), 'Class'+str(classes[1])])
    plt.show()
                                                         
           
                
features=np.array([[2,1],[4,5],[5,3],[3,2]])      
print(features)
labels=np.array([1,0,0,1])           
print(labels)
classes=[0,1]
plot_fun(features,labels,classes)

bias=np.ones((features.shape[0],1))
print(bias)
print(bias.shape)
features=np.append(bias,features,axis=1)
print('Features vector after adding the bias')
print(features)
print(features.shape)        
    
neural_network=NeuralNetwork()
print('Random weights at the start of training')
print(neural_network.weight_matrix)
num_iterations=100
neural_network.train(features,labels,num_iterations)

print('New weights after training')
print(neural_network.weight_matrix)

print('Testing network on training data points -->')
print(neural_network.pred(features))

print('Testing network on new examples-->')
print(neural_network.pred(np.array([[2,2,1],[-2,-2,0],[0,0,0],[-2,0,0]])))
