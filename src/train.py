from conventional.classification.knn import train_knn
from conventional.classification.svm import train_svm
from constant import DATASET_PATH


if __name__ == "__main__":
  print(f"[TRAINING] Start training SVM model")
  train_svm(DATASET_PATH)

  print(f"[TRAINING] Start training KNN model")
  train_knn(DATASET_PATH)