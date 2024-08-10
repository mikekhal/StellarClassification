# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 20:24:44 2024

@author: mike_
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import scipy.stats
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def gen_data(n_samples, n_features, noise):
    # Generate random feature data
    X = np.random.rand(n_samples, n_features)
    
    # Generate an ideal relationship between X and y
    # For example, a linear relationship with some coefficients
    coeffs = np.random.rand(n_features, 1)
    y_ideal = X.dot(coeffs).flatten()  # Ideal target values without noise
    
    # Add noise to the y_ideal to create y
    y = y_ideal + noise * np.random.randn(n_samples)
    
    return X, y, X, y_ideal


df = pd.read_csv('star_dataset.csv', delimiter=',')

selected_columns = df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star type', 'Star color', 'Spectral Class']]
temperature = selected_columns['Temperature (K)']
luminosity = selected_columns['Luminosity(L/Lo)']
radius = selected_columns['Radius(R/Ro)']
absolute_magnitude = selected_columns['Absolute magnitude(Mv)']
star_type = selected_columns['Star type']
star_color = selected_columns['Star color']
spectral_class = selected_columns['Spectral Class']

# Plot Temperature vs Luminosity
plt.figure(figsize=(10, 6))
plt.scatter(df['Temperature (K)'], df['Luminosity(L/Lo)'], alpha=0.5)
plt.xlabel('Temperature (K)')
plt.ylabel('Luminosity(L/Lo)')
plt.title('Temperature vs Luminosity')
plt.grid(True)
plt.show()
# Plot Temperature vs Luminosity
plt.figure(figsize=(10, 6))
plt.scatter(temperature, absolute_magnitude, alpha=0.5)
plt.xlabel('Temperature (K)')
plt.ylabel('Absolute Magnitude (Mv)')
plt.title('Temperature vs Absolute Magnitude (Mv)')
plt.grid(True)
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()

plt.show()
selected_columns['Star color'] =selected_columns['Star color'].str.lower()
selected_columns['Star color'] = selected_columns['Star color'].str.replace('-', ' ')


#relation = df.corr()
le = LabelEncoder()
selected_columns['Star color_encoded']  = le.fit_transform(selected_columns['Star color'])
selected_columns['Spectral Class_encoded'] = le.fit_transform(spectral_class)
X = selected_columns.drop(columns=['Star type', 'Star color', 'Spectral Class'])
y = selected_columns['Star type']
print(type(X), type(y))

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=10)

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=10)
dt = dt.fit(X_train,y_train)
y_test_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_test_pred))
#print(confusion_matrix(y_test, y_test_pred))


from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus
from six import StringIO

dot_data = StringIO()
export_graphviz(dt, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = X.columns,class_names=['0','1','2','3','4','5'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
downloads_path = r'C:\Users\mike_\Downloads\star.png'

# Create and save the PNG file
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(downloads_path)
graph.write_png('star.png')
Image(graph.create_png())

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=10)
rf.fit(X_train, y_train)
y_test_pred_rf = rf.predict(X_test)

# Accuracy and classification report for Random Forest
accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf:.2f}')
print(classification_report(y_test, y_test_pred_rf))

#generate data
#X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)

from sklearn.metrics import mean_squared_error

# Training and Test Errors for Decision Tree
y_train_pred_dt = dt.predict(X_train)
train_error_dt = mean_squared_error(y_train, y_train_pred_dt)

y_test_pred_dt = dt.predict(X_test)
test_error_dt = mean_squared_error(y_test, y_test_pred_dt)

print(f'Decision Tree - Training Error: {train_error_dt:.2f}')
print(f'Decision Tree - Test Error: {test_error_dt:.2f}')

# Training and Test Errors for Random Forest
y_train_pred_rf = rf.predict(X_train)
train_error_rf = mean_squared_error(y_train, y_train_pred_rf)

y_test_pred_rf = rf.predict(X_test)
test_error_rf = mean_squared_error(y_test, y_test_pred_rf)

print(f'Random Forest - Training Error: {train_error_rf:.2f}')
print(f'Random Forest - Test Error: {test_error_rf:.2f}')

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
    )

    train_errors_mean = -np.mean(train_scores, axis=1)
    valid_errors_mean = -np.mean(valid_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_errors_mean, label='Training error')
    plt.plot(train_sizes, valid_errors_mean, label='Validation error')
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.title('Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot learning curves for Decision Tree
plot_learning_curve(dt, X, y)

# Plot learning curves for Random Forest
plot_learning_curve(rf, X, y)
