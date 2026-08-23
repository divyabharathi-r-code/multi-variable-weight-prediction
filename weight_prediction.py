import numpy as np
from sklearn.linear_model import LinearRegression

X= np.array([[165,21.3],[168,21.6],[170,22.1],[172,22.6],[174,21.8]])
y= np.array([58,61,64,67,66])
model = LinearRegression()
model.fit(X,y)
height_bmi_totest = [186,22.3]
unknown_person= np.array([height_bmi_totest]) 
prediction = model.predict(unknown_person)
print(f"Prediction of weight{prediction[0]:.2f}")
accuracy = model.score(X, y)
print(f"R2 Score : {accuracy * 100:.2f}%")