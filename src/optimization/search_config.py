from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

param_grids = {
    'Logistic Regression': {'classifier__C': [0.1, 1, 10]},
    'SVM (Linear)': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['linear']},
    'SVM (RBF)': {'classifier__C': [0.1, 1, 10], 'classifier__kernel': ['rbf']},
    'Decision Tree': {'classifier__max_depth': [None, 10, 20]},
    'Random Forest': {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]},
    'Gradient Boosting': {'classifier__learning_rate': [0.01, 0.1], 'classifier__n_estimators': [50, 100]},
    'KNN': {'classifier__n_neighbors': [3, 5, 10]},
    'Neural Network': {'classifier__hidden_layer_sizes': [(50,), (100,)], 'classifier__alpha': [0.0001, 0.001]}
}

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'SVM (Linear)': SVC(probability=True, random_state=42),
    'SVM (RBF)': SVC(probability=True, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'KNN': KNeighborsClassifier(),
    'Neural Network': MLPClassifier(max_iter=10000, random_state=42)
}
