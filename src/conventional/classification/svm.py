from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from conventional.load_dataset import load_dataset
from conventional.preprocessing import preprocess_image
import time

def svm(dataset_path, img):
    try:
        features, labels = load_dataset(dataset_path)
        print(f"Features shape: {features.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None, None

    # Feature scaling
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # Hyperparameter tuning
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'degree': [2, 3, 4]
    }
    grid = GridSearchCV(SVC(probability=True), param_grid, cv=10, scoring='accuracy')
    start_time = time.time()
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    svm = grid.best_estimator_
    print(f"Training Time: {time.time() - start_time:.2f} seconds")

    # Evaluation
    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))

    # Preprocess new image
    new_features = preprocess_image(img.convert('RGB'), features.shape[1]).reshape(1, -1)
    new_features = scaler.transform(new_features)

    # Predict new image
    proba = svm.predict_proba(new_features)
    prediction = svm.predict(new_features)
    print(f"Predicted Class: {prediction[0]}, Probabilities: {proba[0]}")

    return prediction, proba