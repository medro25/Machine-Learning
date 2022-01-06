import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
# read the data using pandas library
data =pd.read_csv("student-mat.csv", sep=";")
# categorize data
data = data[["G1","G2","G3","studytime","failures","absences"]]
print(data.head(5))
# our main target is to predict grade 3 of students
predict="G3"

# set up the attributes(features) in array /
# using the drop function to  remove column of grade 3 from the DataFrame which are specify in first parameter labels
x= np.array(data.drop([predict], 1))
# set up the labels in an array
y=np.array(data[predict])
# dividing the data into test and train data to test our accuracy of our model
x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(x,y, test_size=0.1)
#implementing our training  model ( algorithm)
linear=linear_model.LinearRegression()
#find the best fit to the data using X train data and y train data
linear.fit(x_train,y_train)
# test our model : how well is working
acc=linear.score(x_test,y_test)
print(acc)
# print out the coefficiant and the intercept
# we will get 5 coefficiant since we have 5 attributes ( 5 dimentions )
print("coeff: \n",linear.coef_)
print("intercept: \n",linear.intercept_)
prediction =linear.predict(x_test)
#print("prediction:\n",prediction)
for x in range (len(prediction)):
    print((prediction[x],x_test[x],y_test[x]))