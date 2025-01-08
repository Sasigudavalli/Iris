# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:31:15 2024

@author: user
"""

import streamlit as st
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

data = sns.load_dataset("iris")
data

data["species"].unique()
data["species"]=data["species"].replace({'setosa':0, 'versicolor':1, 'virginica':2})
X=data.iloc[:,0:4]
y=data["species"]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(solver='liblinear', multi_class='ovr'),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(gamma='auto')
}

kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
st.title("Machine Learning Model Selector")
st.write("Choose a model and evaluate its performance.")


model_name = st.selectbox("Select a Model", list(models.keys()))


if st.button("Evaluate Model"):

    model = models[model_name]
    

    
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    
    
    st.write(f"Model: {model_name}")
    st.write(f"Accuracy: {cv_results.mean():.4f} ")


if st.button("Find Best Model"):
    best_model_name = None
    best_score = 0.0
    
    for name, model in models.items():
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
        mean_score = cv_results.mean()
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
    
    st.write(f"The best model is: **{best_model_name}** with an accuracy of **{best_score:.4f}**.")