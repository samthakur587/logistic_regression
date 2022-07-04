# logistic_regression
I have created the logistic regression with a neural network mindset.
and also i have created a file colled logistic_reg_method.py in which i 
have explained the method that we can use in this model.

# Perameters that are used in this model.
I have used the 5 perameters in it (bd -> by default)
>1) Learning_rate [Range 0.00001 - 1]  bd -> 0.005

this perameter play a very effective role the logistic regression. it is used to reduse the cost of an
function . the value of learning_rate indecates that how bit step we are going towards the minimum of that function
w.r.t number of iteration.

>2) num_iteration [Range >100] bd -> 1000

this perameter used to iterate the model to fit perfect to the data . the high number of itration can get a good accuracy in training 
set . but the high number of iteration may be leads to overfit problem in training set.

>3) print_cost [True,False] bd -> False

this peramters is very importent for debuging the model . it will show you the cost of the function per hundred iteration. so you can tune your model
more better then before.

>4) regularization [True ,False] bd -> True

this perameter is used if we want to do regularization or not if do then set this perameter as True otherwise False

>5) lambdaa [0.01-100] bd -> 10

the is basically used as a wight of the regularization how much regularization do you need in your model

[License](LICENSE)

[Linkedin](https://www.linkedin.com/in/samunder-singh-265508202?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BD2%2BZpxmSTzmXNjBgIdoUSA%3D%3D)
