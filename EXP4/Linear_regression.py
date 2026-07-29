import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
datr = pd.read_csv(r"C:\Users\panka\OneDrive\Desktop\ACADEMICS\Dipika Maam\AIML\EXP4\data.csv")
x = datr[['YearsExperience']]
y = datr['Salary']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = LinearRegression()  
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
# print(y_pred)
experience = float(input("Enter the years of experience: "))    
predicted_salary = model.predict([[experience]])
# print(f"The predicted salary for {experience} years of experience is: {predicted_salary}")
plt.scatter(x,y,color = 'blue' ,label = 'Actual Data')
plt.scatter(experience,predicted_salary,color = 'green')
plt.plot(x,model.predict(x),color = 'red',label = 'Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Salary vs Years of Experience')  
plt.legend()
plt.show()
