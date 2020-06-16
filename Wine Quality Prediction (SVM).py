from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv("winequality-red.csv",delimiter = ';')
#print(df.head())

correlations = df.corr()['quality'].drop('quality')
#print(correlations)

bins = (2,6.5,8)
group_names = ['bad','good']
categories = pd.cut(df['quality'], bins, labels = group_names)
df['quality'] = categories

x = df.drop(['quality'], axis = 1)
y = df.quality

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state= 6)

sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

classifier= SVC(C = 10,gamma= 0.5)
classifier.fit(x_train,y_train)

predictions = classifier.predict(x_test)
print(accuracy_score(y_test, predictions))

parameters = [{'C': [1, 10, 100], 'kernel': ['linear']},
              {'C': [1, 10, 100], 'kernel': ['rbf'],
               'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print(best_accuracy)
print(best_parameters)

