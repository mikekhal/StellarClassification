# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 21:24:31 2024

@author: mike_
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image
import pydotplus
from six import StringIO
import numpy as np
from scipy.stats import spearmanr

 # Load dataset
df = pd.read_csv('star_dataset.csv', delimiter=',')
 
def preprocess_data(df):
     selected_columns = df[['Temperature (K)', 'Luminosity(L/Lo)', 'Radius(R/Ro)', 'Absolute magnitude(Mv)', 'Star type', 'Star color', 'Spectral Class']]
     
     # Normalising data in 'Star color' heading
     selected_columns['Star color'] = selected_columns['Star color'].str.lower().str.replace('-', ' ')

     # Encode categorical features
     le = LabelEncoder()
     selected_columns['Star color_encoded'] = le.fit_transform(selected_columns['Star color'])
     selected_columns['Spectral Class_encoded'] = le.fit_transform(selected_columns['Spectral Class'])
     selected_columns['Star type_encoded'] = le.fit_transform(selected_columns['Star type'])
     
     X = selected_columns.drop(columns=['Star type', 'Star color', 'Spectral Class', 'Star type_encoded']) # features
     y = selected_columns['Star type'] # target value
     
     return selected_columns, X, y
selected_columns, X, y = preprocess_data(df)

# Plot pair plot
def plot_pairplot(data):
    sns.pairplot(data, hue='Star type_encoded')
    plt.suptitle('Pair Plot of Star Dataset', y=1.02)
    plt.show()
plot_pairplot(selected_columns)

# Find correlation between variables (spearman's correlation)
def calculate_spearman_correlations(data):
    columns = data.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            corr, _ = spearmanr(data[columns[i]], data[columns[j]])
            print(f"Spearman's correlation between {columns[i]} and {columns[j]}: {corr:.3f}")
calculate_spearman_correlations(selected_columns)

# Plot HR diagram
def plot_hr_diagram(data):
    # Extracting necessary columns for the HR diagram
    temperature = np.log(data['Temperature (K)'])
    absolute_magnitude = np.log(data['Absolute magnitude(Mv)'])
    star_type = data['Star type']
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(temperature, absolute_magnitude, c=star_type, cmap='viridis', edgecolor='k', alpha=0.7)
    
    plt.gca().invert_xaxis()
    
    plt.colorbar(scatter, label='Star Type')
    plt.title('Hertzsprung-Russell Diagram')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Absolute Magnitude (Mv)')
    plt.grid(True)
    plt.show()

plot_hr_diagram(selected_columns)

def plot_learning_curve(model, X, y):
    """
    Plot learning curves for the given model.

    Parameters:
    - model: The machine learning model
    - X: Feature matrix
    - y: Target vector
    """
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

def visualize_decision_tree(model, X):
    """
    Visualize the Decision Tree.

    Parameters:
    - model: The trained Decision Tree model
    - X: Feature matrix used for training the model
    """
    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True, feature_names=X.columns, class_names=['0','1','2','3','4','5'])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    png_path = 'star.png'
    graph.write_png('star.png')
    Image(graph.create_png())
    # Save and display the Decision Tree visualization
    graph.write_png(png_path)
    Image(graph.create_png())

# Using Decision Tree Classifier algorithm
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=10)
dt.fit(X_train, y_train)
y_test_pred_dt = dt.predict(X_test)
print(f'Decision Tree - Accuracy: {accuracy_score(y_test, y_test_pred_dt):.2f}')
print(f'Decision Tree - Classification Report:\n{classification_report(y_test, y_test_pred_dt)}')
print(f'Decision Tree - Training Error: {mean_squared_error(y_train, dt.predict(X_train)):.2f}')
print(f'Decision Tree - Test Error: {mean_squared_error(y_test, y_test_pred_dt):.2f}')

# Train and evaluate Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=10)
rf.fit(X_train, y_train)
y_test_pred_rf = rf.predict(X_test)

print(f'Random Forest - Accuracy: {accuracy_score(y_test, y_test_pred_rf):.2f}')
print(f'Random Forest - Classification Report:\n{classification_report(y_test, y_test_pred_rf)}')
print(f'Random Forest - Training Error: {mean_squared_error(y_train, rf.predict(X_train)):.2f}')
print(f'Random Forest - Test Error: {mean_squared_error(y_test, y_test_pred_rf):.2f}')

# Plot learning curves
plot_learning_curve(dt, X, y)
plot_learning_curve(rf, X, y)

# Visualize Decision Tree
visualize_decision_tree(dt, X)