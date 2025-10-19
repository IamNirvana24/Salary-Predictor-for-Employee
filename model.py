#Step 1- Import Librariers
import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#Uploading Data for Analysis
data = pd.read_csv(r"C:\Machine Learning\Projects\Machine Learning Models\Logistic & Linear Regression\Salary Predictor\salary_numeric_dataset.csv")
print(data.head)
#Model Training
X = data[['Experience (Years)', 'Skill Score', 'Age']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state= 42) 


model = LinearRegression()
model.fit(X_train, y_train)
#Model Prediction
y_pred = model.predict(X_test)


print("Actual", list(y_test[0:8]))
print("Predicted", list(y_pred[0:8]))

#Data Visuallization
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted Salary")
plt.show()