import os
import pickle
from constant import DATASET_PATH
from conventional.load_dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

knn_model_file_path = os.path.join(os.getcwd(), 'src', 'conventional', 'model', 'knn_model.pkl')
svm_model_file_path = os.path.join(os.getcwd(), 'src', 'conventional', 'model', 'svm_model.pkl')

if __name__ == "__main__":
  with open(knn_model_file_path, 'rb') as f:
      knn = pickle.load(f)  

  with open(svm_model_file_path, 'rb') as f:
      svm = pickle.load(f)

  features, labels = load_dataset(DATASET_PATH)
  _, X_test, _, y_test = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=42
    )
  
  print("KNN Report")
  knn_pred = knn.predict(X_test)
  print(classification_report(y_test, knn_pred, zero_division=0))

  print("SVM Report")
  svm_pred = svm.predict(X_test)
  print(classification_report(y_test, svm_pred, zero_division=0))
