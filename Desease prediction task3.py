from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

X = data.data
y = data.target

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier()

model.fit(X_train,y_train)

prediction = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,prediction))

sample = X_test[0].reshape(1,-1)

result = model.predict(sample)

if result[0] == 1:
    print("Disease Not Detected")
else:
    print("Disease Detected")