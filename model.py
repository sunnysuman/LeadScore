import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df=pd.read_csv('Leads.csv')

df=df[['Total Time Spent on Website','TotalVisits','Page Views Per Visit','Converted']]

df.fillna(df.mean())

df=df.dropna(axis=0)

features=df[['Total Time Spent on Website','TotalVisits','Page Views Per Visit']]

labels=df[['Converted']]

from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
features.iloc[:,:] = Scaler.fit_transform(features.iloc[:,:])

#df.isnull().sum()

X_train,x_test,Y_train,y_test=train_test_split(features,labels,test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# Fit the model on training set

# save the model to disk
import pickle

filename = 'model.pkl'
pickle.dump(logreg, open(filename, 'wb'))

# some time later...

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)