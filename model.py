import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

X = df[['study_hours', 'attendance']]
y = df['result']

model = LogisticRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved")
