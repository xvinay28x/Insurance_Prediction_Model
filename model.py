import pandas as pd 
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("insurance_data(with_outlier).csv")

x = df[["age"]]
y = df["bought_insurance"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

model = LogisticRegression()
model.fit(x_train,y_train)

x=model.predict([[19]])
print(x)

pickle.dump(model,open("Pickle_file.pkl","wb"))
