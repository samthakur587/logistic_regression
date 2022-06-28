import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
class neuralnetwork_logisticregression():
    def __init__(self,num_iteration=1000,learning_rate=0.005,print_cost=False ,lambdaa=10,regularization=True):
        self.num_iteration = num_iteration
        self.learning_rate = learning_rate
        self.print_cost = print_cost
        self.lambdaa = lambdaa
        self.regularization = regularization
    def sigmoid(self,x):
        s = 1/(1+np.exp(-x))
        return s
    def propagate(self,x,y,w,b):
        eps = 0.0000001
        grads = {}
        A = self.sigmoid(np.dot(w.T , x)+ b)
        cost = -(1/self.m)*np.sum(y*np.log(A+eps) + (1-y)*np.log(1-A+eps),axis=1)
        #cost =  (1/self.m)*np.sum(abs(A-y))
        cost = np.squeeze(np.array(cost))
        dw = (1/self.m)*np.dot(x,(A-y).T)
        db = (1/self.m)*np.sum((A-y))
        if self.regularization :
            l2_reg = (self.lambdaa/(2*self.m))*np.sum(w**2)
            cost = cost + l2_reg
            dw = dw + (self.lambdaa/self.m)*w
        grads['dw'] = dw
        grads['db'] = db
        return cost , grads
    def optimize(self,x,y,w,b):
        self.costs = []
        perms = {}
        for i in range(self.num_iteration):
            cost , grads = self.propagate(x,y,w,b)
            dw = grads['dw']
            db = grads['db']
            w = w - self.learning_rate*dw
            b = b - self.learning_rate*db
            if i%100 ==0 :
                self.costs.append(cost)
                if self.print_cost:
                    print ("Cost after iteration %i: %f\n" %(i, cost))
        perms['w'] = w
        perms['b'] = b

        return perms
    def fit(self,x,y):
        w = np.zeros((x.shape[0],1))
        b = float(0)
        self.m = x.shape[1]
        self.perms = self.optimize(x,y,w,b)
        return 0
    def cost_return(self):
        return self.costs
    def predict(self,x):
        i=0
        w = self.perms['w']
        b  = self.perms['b']
        m = x.shape[1]
        y_pred = np.zeros((1,m))
        A = self.sigmoid(np.dot(w.T,x) + b)
        for i in range(m):

            if A[0,i] > 0.5:
                y_pred[0,i] = 1
            else :
                y_pred[0,i] = 0
        return y_pred
    def model_report(self,y_pred , y_test):
        tp=0
        tn=0
        fp=0
        fn=0
        report = {}
        for i in range(50):
            if y_pred[0][i] ==1.0 and y_test[0][i] ==1 :

                tp = tp+1
            if y_pred[0][i] ==1.0 and y_test[0][i] ==0 :
                fn = fn+1
            if y_pred[0][i] ==0.0 and y_test[0][i] ==1 :
                fp = fp+1
            if y_pred[0][i] ==0.0 and y_test[0][i] ==0 :
                tn = tn+1

        confusion_matrix = np.array([[tp,fp],
                                    [fn , tn]])
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        Accuracy = (tp+tn)/(tp+tn+fp+fn)
        f1 = 2*recall*precision/(precision+recall)
        print("\nconfusion_matrix : \n",confusion_matrix)

        report['precision'] = str(precision*100) + "%"
        report['recall'] = str(recall*100) + "%"
        report['Accuracy'] = str(Accuracy*100)+"%"
        report['f1_score'] = str(f1*100) + "%"
        data = pd.DataFrame(list(report.items()), columns = ['report', 'value'])
        print("\nreport\n",data)
    def plotcost(self):
        plt.plot(self.costs,color='g')

        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(self.learning_rate))
        plt.show()
        return 0
    def load_dataset(self,files):
        hf = h5py.File(files,'r')
        key = list(hf.keys())
        n1 = np.array(hf.get(key[1]))
        n2 = np.array(hf.get(key[2]))
        n1 = n1.reshape(n1.shape[0],-1).T

        n2 = n2.reshape(1,n2.shape[0])
        return n1/255 , n2
def main():
    model = neuralnetwork_logisticregression(num_iteration = 6000,learning_rate=0.001,print_cost=True,lambdaa=11 )
    x_train , y_train = model.load_dataset('train_catvnoncat.h5')
    x_test , y_test = model.load_dataset('test_catvnoncat.h5')
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print(model.model_report(y_pred,y_test))
    model.plotcost()


if __name__ == "__main__":
    main()
