import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample Dataset
data = {
    'Income': [30000,40000,50000,60000,70000,80000,90000,100000],
    'Debt': [5000,10000,7000,2000,15000,5000,10000,2000],
    'CreditScore': [600,650,700,750,680,720,760,800],
    'Approved': [0,0,1,1,0,1,1,1]
}

df = pd.DataFrame(data)

X = df[['Income','Debt','CreditScore']]
y = df['Approved']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, prediction))

# User Input
income = int(input("Enter Income: "))
debt = int(input("Enter Debt: "))
score = int(input("Enter Credit Score: "))

result = model.predict([[income,debt,score]])

if result[0] == 1:
    print("Credit Approved")
else:
    print("Credit Rejected")