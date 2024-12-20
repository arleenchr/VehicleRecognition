from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from conventional.load_dataset import load_dataset
from conventional.preprocess_image import preprocess_image
import time
import os
import pickle
from constant import DATASET_PATH

# KNN
knn_model_file_path = os.path.join(os.getcwd(), 'src', 'conventional', 'model', 'knn_model.pkl')

def train_knn(dataset_path):
    features, labels = load_dataset(dataset_path)

    # Feature scaling
    # scaler = StandardScaler()
    # features_scaled = scaler.fit_transform(features)
    # print("Features scaled.")

    # # Dimensionality reduction with PCA
    # pca = PCA(n_components=0.95)  # Retain 95% variance
    # features_reduced = pca.fit_transform(features_scaled)
    # print("Features reduced to shape:", features_reduced.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
            
    print("Training set size:", X_train.shape[0])
    print("Test set size:", X_test.shape[0])

    # Hyperparameter tuning for KNN using GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to test
        'weights': ['uniform', 'distance'],  # Weighting method for neighbors
        'metric': ['euclidean', 'manhattan', 'minkowski']  # Distance metrics
    }

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, n_jobs=-1)
    
    # Start timing the grid search
    start_time = time.time()
    grid.fit(X_train, y_train)
    print(f"Best Parameters: {grid.best_params_}")
    knn = grid.best_estimator_
    print(f"GridSearchCV Time: {time.time() - start_time:.2f} seconds")

    # Evaluate the model
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=0))
    print(confusion_matrix(y_test, y_pred))

    # Save Model
    with open(knn_model_file_path, 'wb') as f:
        pickle.dump(knn, f)

def predict_knn(img):
    feature, bounded_image = preprocess_image(img, target_feature_size=1000)
    
    # Load Model
    with open(knn_model_file_path, 'rb') as f:
      knn = pickle.load(f)

    # Preprocess new image
    feature = feature.reshape(1, -1)
    # scaler = MinMaxScaler()
    # feature = scaler.fit_transform(feature)
    
    # # Scale and reduce the new features
    # new_features = scaler.transform(new_features)
    # print("New features shape SCALER:", new_features.shape)
    # new_features = pca.transform(new_features)
    # print("New features shape PCA:", new_features.shape)
    
    # Predict new image
    proba = knn.predict_proba(feature)
    prediction = knn.predict(feature)
    classes = knn.classes_
    print(f"Predicted Class: {prediction[0]}, Probabilities: {proba[0]}")

    return prediction, proba, classes, bounded_image