import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import logistic_regrassion as lr
###############################################################
# this you can use for getting the methods from your model
model = lr.neuralnetwork_logisticregression()
method_list = [method for method in dir(model) if method.startswith('__') is False]
print(method_list)
##############################################################
# this can be used to determine the best value for perameter lambdaa amd learning_rate w.r.t minimum cost
lr_costs = {}
beta_list = [0.01,0.1,1,10,100]
for beta in beta_list :
    model = lr.neuralnetwork_logisticregression(lambdaa = beta)
    x_train , y_train = model.load_dataset('data/train_catvnoncat.h5')
    x_test , y_test = model.load_dataset('data/test_catvnoncat.h5')
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    model.model_report(y_pred,y_test)
    cost = model.cost_return()
    lr_costs[beta] = cost
data = pd.DataFrame(lr_costs)
print(data)
for beta in beta_list:
    plt.plot(lr_costs[beta], label = "regularization_constant" + str(beta))
    plt.legend()
plt.show()
#################################################################
# describe all the methodes that are used in this model

#1) load_dataset('file.h5'):
#           it is methode that used to load the training and testing data.
# it takes the .h5 file as input (but its only take .h5 file which has the
# keys() in ['label','x_value','y_value']) it is best for my data you can use
#cv2 for data loding into training and testing set but the dimentions of you
# x_set is (row , num_of_example) , or y_set is (1,num_of_example).

#2) fit(x_train,y_train ):
#         it is used to train your model and get the best perameter (w,b) for
#         your model

#3) predict(x_test):
#           it is used to predict the y_pred value for your x_test

#4) model_report(y_pred,y_test) :
#        this method gives you the all overview about your model such as
#        precision , recall , accuracy , f1 score and confusion_matrix
#        so that you know very well about your model

#5) plotcost() :
#               from this method you can directily plot the cost value weather it is decresing or incresing
#               so you can determine the correct value of lambdaa and learning_rate.
#6) cost_return():
#                this is used for if you want to return the cost value for some fix perameter
