# 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# ---------------------------
# Step 1: Load Dataset
# ---------------------------
df = pd.read_csv('iris.csv')  # replace with your CSV path
print("First 5 rows of dataset:")
print(df.head())

# ---------------------------
# Step 2: Prepare Features & Labels
# ---------------------------
X = df.iloc[:, :-1].values  # Sepal & Petal lengths and widths
y = df.iloc[:, -1].values   # Species column

# Encode species names to numbers
le = LabelEncoder()
y = le.fit_transform(y)  # Setosa=0, Versicolor=1, Virginica=2

# ---------------------------
# Step 3: Split Data (80-20)
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Step 4: Train SVM Model
# ---------------------------
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# ---------------------------
# Step 5: Test & Accuracy
# ---------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on test set: {accuracy*100:.2f}%")

# ---------------------------
# Step 6: Visualize Data
# ---------------------------
species_names = le.inverse_transform(np.unique(y))

for i, s in enumerate(np.unique(y)):
    plt.scatter(X[y == s, 0], X[y == s, 1], label=species_names[i])

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset')
plt.legend()
plt.show()

# ---------------------------
# Step 7: Predict New Flower
# ---------------------------
# Example input: Sepal L=5, Sepal W=3, Petal L=1.5, Petal W=0.2
new_flower = [[5, 3, 1.5, 0.2]]
predicted = model.predict(new_flower)
print("Predicted species for new flower:", le.inverse_transform(predicted)[0])
