import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('framingham.csv')
df.head()

df.dropna(axis=0, inplace=True)
print(df.shape)

df['TenYearCHD'].value_counts()


plt.figure(figsize = (14, 10))
sns.heatmap(df.corr(), cmap='Purples',annot=True, linecolor='Green', linewidths=1.0)
plt.show()


X = df.iloc[:,0:15]
y = df.iloc[:,15:16]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
print(X_train)
print(y_train)

logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
score = logreg.score(X_test, y_test)
print("Prediction score of the trained model is:",score)


from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of model is:\n",cm)

print("Classification Report of model is:\n\n", classification_report(y_test, y_pred))
conf_matrix = pd.DataFrame(data = cm,
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])
plt.figure(figsize = (10, 6))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "Greens", linecolor="Blue", linewidths=1.5)
plt.show()
