from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from conventional.load_dataset import load_dataset
from conventional.preprocess_image import preprocess_image
import time
import os
import pickle

# SVM
svm_model_file_path = os.path.join(os.getcwd(), 'src', 'conventional', 'model', 'svm_model.pkl')

def train_svm(dataset_path: list, amount_each_class: int = 200):
    features, labels = load_dataset(dataset_path, amount_each_class)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4]
    }
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=10, scoring='accuracy', n_jobs=-1)
    start_time = time.time()
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    svm = grid.best_estimator_
    print(f"Training Time: {time.time() - start_time:.2f} seconds")

    # Evaluation
    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))

    # Save Model
    with open(svm_model_file_path, 'wb') as f:
        pickle.dump(svm, f)


def predict_svm(img):
    feature, bounded_image = preprocess_image(img, target_feature_size=1000)
    
    # Load Model
    with open(svm_model_file_path, 'rb') as f:
      svm = pickle.load(f)

    # Preprocess new image
    feature = feature.reshape(1, -1)

    # Predict new image
    proba = svm.predict_proba(feature)
    prediction = svm.predict(feature)
    classes = svm.classes_
    print(f"Predicted Class: {prediction[0]}, Probabilities: {proba[0]}")

    return prediction, proba, classes, bounded_image