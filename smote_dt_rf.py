from xml.etree.ElementInclude import include
from IPython.display import display
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Data Preprocessing
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Model ML
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

# Evaluasi
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score, accuracy_score, precision_score, recall_score

#Data Loading
df = pd.read_csv('stroke.csv')
# # 열 이름 추출
# column_names = df.columns.tolist()
# # 추출된 열 이름을 csv 파일로 저장
# pd.DataFrame(column_names).to_csv('column_names.csv', index=False, header=False)

df.head()
df.info()
df.isnull().sum()

#EDA(Exploratory Data Analysis)
df.drop('id', inplace=True, axis=1)
print("Summary statistics:")
display(df.describe(include='number'))

print("Class distribution:")
display(df[['work_type','gender']].value_counts())
df.drop(df[df['gender']== 'Other'].index, inplace=True)

#Handling Missing Values
df.isnull().sum()
# Checking the distribution of BMI for determining BMI fillna strategy
sns.histplot(data = df, x='bmi',kde=True);
plt.show()
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())
df.isnull().sum()

#Plotting each column in the dataset
fig, ax = plt.subplots(3,4)
fig.set_size_inches(30, 20)

features = ['age', 'avg_glucose_level', 'bmi', 'gender', 'hypertension', 'heart_disease',
            'ever_married', 'work_type', 'Residence_type', 'smoking_status','stroke']
numerical_features = ['age', 'avg_glucose_level', 'bmi']

for i, feature in enumerate(features):
    row = i // 4
    col = i % 4
    if feature in numerical_features:
        hist = sns.histplot(data=df, x=feature, ax=ax[row, col], kde=True)
    else:
        hist = sns.countplot(data=df, x=feature, ax=ax[row, col], hue=feature)
    hist.set_title(feature, fontsize=20)
    hist.set_xlabel('')

plt.show()

strok=df[df['stroke']==1]
sns.countplot(data=strok, x='ever_married', hue='ever_married')
plt.title('Number of Stroke Cases Based on Marital Status')

# Filtering the data for males and females who experienced a positive stroke.
men_positive = df[(df['stroke'] == 1) & (df['gender'] == 'Male')]
women_positive = df[(df['stroke'] == 1) & (df['gender'] == 'Female')]

# Plotting
plt.figure(figsize=(10, 4))

# histogram plot for men
plt.subplot(1, 2, 1)
plt.hist(men_positive['age'], bins=np.linspace(0, 80, 7), color='blue')
plt.xlabel('Umur')
plt.ylabel('Frequencies')
plt.title('Positive Stroke in Males')
plt.ylim(0, 100)

# histogram plot for women
plt.subplot(1, 2, 2)
plt.hist(women_positive['age'], bins=np.linspace(0, 80, 5), color='pink')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Positive Stroke in Woman')
plt.ylim(0, 100)

plt.tight_layout()
plt.show()

# Calculating the number of strokes (value 1) in the 'stroke' column.
number_of_strokes = df['stroke'].value_counts()[1]

# Calculating total row of dataframe
total_data = df.shape[0]

# Calculating stroke precentation
stroke_percentage = (number_of_strokes / total_data) * 100

print("Perception of stroke patient: {:.2f}%".format(stroke_percentage))

total_data
# Correlation among numerical data
numerical_df = df.select_dtypes(exclude = 'object')
corr = numerical_df.corr()
sns.heatmap(corr, annot=True, mask = np.triu(np.ones_like(corr, dtype=bool)))

#Data preprocessing
print(df.columns.get_loc('work_type'))
# Encoding Categorical Data with One-Hot Encoding
df_onehot = pd.get_dummies(df).astype(int)
df_onehot.head(3)
df_onehot.columns

# Separate df_onehot to be X and y
X = df_onehot.drop(columns=['stroke'])  # Features
y = df_onehot['stroke']  # Target variable

# Split the data for train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8 , random_state = 123)

y_test.value_counts()
X_train.columns

# To find columns in X_train that contain boolean values
cf = X_train.apply(lambda col: col.isin([0, 1]).all())

# Index of columns containing boolean data
cat_features_col=[]

for key, value in cf.items():
    if value:
      cat_features_col.append(X_train.columns.get_loc(str(key)))

cat_features_col

from imblearn.over_sampling import SMOTENC

# SMOTENC
smotenc = SMOTENC(categorical_features= cat_features_col, k_neighbors = 5, random_state = 456)

# Oversampling on X_train and y_train
X_train_balanced, y_train_balanced = smotenc.fit_resample(X_train, y_train)
y_train_balanced.value_counts()

#Modeling
#DT AND RF

#DT
dt = DecisionTreeClassifier(random_state=78)
dt.fit(X_train_balanced, y_train_balanced)
print("Default Parameters for Decision Tree:")
print(dt.get_params())

# Model Prediction

y_pred_dt = dt.predict(X_test)
print(classification_report(y_test, y_pred_dt))
print(ConfusionMatrixDisplay.from_estimator(dt, X_test, y_test, cmap = 'Reds'))

#RF
rf = RandomForestClassifier(random_state=29)
rf.fit(X_train_balanced, y_train_balanced)

rf_base_params = rf.get_params()
for key, value in rf_base_params.items():
  print(f"{key} : {value}")

# Random Forest Model Evaluation
y_pred_rf = rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
print(ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test, cmap='Reds'))
accuracy = accuracy_score(y_test, y_pred_rf)
print(accuracy)

#Cross Validate
cv_dt = cross_validate(dt, X_train_balanced, y_train_balanced, cv=5, scoring=['accuracy', 'recall', 'f1'])
cv_rf = cross_validate(rf, X_train_balanced, y_train_balanced, cv=5, scoring=['accuracy', 'recall', 'f1'])
print(cv_dt)
print(cv_rf)

#Hyperparameter Tuning
#Random Search
n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num = 8)]
max_depth = [int(x) for x in np.linspace(10, 1000, 8)]
min_samples_split = [2, 5, 10, 14,17,22,26]
min_samples_leaf = [1, 2, 4, 6, 8, 10, 14, 16]
params_rf = {'n_estimators': n_estimators,
             'max_depth':max_depth,
             'min_samples_split':min_samples_split,
             'min_samples_leaf':min_samples_leaf}
# Train with Random Search
rf_randomcv = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                 param_distributions=params_rf,
                                 n_iter=10,
                                 cv=5,
                                 random_state = 27,
                                 n_jobs=-1, #parallel processing
                                 scoring='f1',
                                 verbose=3)
rf_randomcv.fit(X_train_balanced, y_train_balanced)
# Get Best Hyperparameters
rf_randomcv.best_params_
rf_randomcv_tuned = rf_randomcv.best_estimator_
rf_randomcv_tuned
# Performance checking of RF model on Test-Set

y_pred_rf_tuned = rf_randomcv_tuned.predict(X_test)

print('F1 Score - Test Set   : ', f1_score(y_test, y_pred_rf_tuned), '\n')
print('Classification Report : \n', classification_report(y_test, y_pred_rf_tuned), '\n')
print('Confusion Matrix      : \n', ConfusionMatrixDisplay.from_estimator(rf_randomcv_tuned, X_test, y_test, cmap='Reds'))

#Grid Search
n_estimators = [1,10,100,150]
max_depth = [int(x) for x in np.linspace(10, 300,5)]
criterion = ['gini','entropy','log_loss']
min_samples_leaf = [1, 2, 4, 6, 15]

params_rf_grid = {'n_estimators': n_estimators,
             'max_depth':max_depth,
             'criterion':criterion,
             'min_samples_leaf':min_samples_leaf}

# Initialization GridSearchCV
rf_gridcv = GridSearchCV(estimator=RandomForestClassifier(),
                         param_grid=params_rf_grid,
                         cv=3,
                         scoring='f1',
                         n_jobs=-1,  # parallel processing
                         verbose=3)

# Perform grid search on balanced data
rf_gridcv.fit(X_train_balanced, y_train_balanced)
# Get Best Hyperparameters
rf_gridcv.best_params_
rf_gridcv_tuned = rf_gridcv.best_estimator_
# Checking Performance of Model RF on Test-Set

y_pred_rf_tuned = rf_gridcv_tuned.predict(X_test)

print('F1 Score - Test Set   : ', f1_score(y_test, y_pred_rf_tuned), '\n')
print('Classification Report : \n', classification_report(y_test, y_pred_rf_tuned), '\n')
print('Confusion Matrix      : \n', ConfusionMatrixDisplay.from_estimator(rf_gridcv_tuned, X_test, y_test, cmap='Reds'))

import joblib

# Save the best model from GridSearchCV to file
joblib.dump(rf_gridcv_tuned, 'random_forest_model.pkl')
# 열 이름 추출 및 CSV로 저장
column_names = X_train.columns.tolist()
pd.DataFrame(column_names).to_csv('column_names.csv', index=False, header=False)