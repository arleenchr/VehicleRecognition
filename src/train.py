from conventional.classification.knn import train_knn
from conventional.classification.svm import train_svm
from deep_learning.model import load_model
from constant import DATASET_PATH


if __name__ == "__main__":
  print(f"[TRAINING] Start training SVM model")
  train_svm(DATASET_PATH[:1])

  print(f"[TRAINING] Start training KNN model")
  train_knn(DATASET_PATH[:1])

  # print(f"[TRAINING] Start training CNN model")
  # load_model(DATASET_PATH[0])