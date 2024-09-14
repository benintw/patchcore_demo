import matplotlib.pyplot as plt
from pathlib import Path
from torchvision.transforms import transforms
import numpy as np


from sklearn.metrics import f1_score
from icecream import ic

import config
from resnet_feature_extractor import ResNetFeatureExtractor
from memory_bank import create_memory_bank
from evaluation import get_y_score_from_training_set, get_y_score_and_true_from_testing
from utils import calculate_roc_auc, plot_roc_curve, plot_confusion_matrix
from prediction import predict_on_test_set

# Settings
SUBJECT = "transistor"
DEFECT_TYPE = "bent_lead"  # can be modified to detect all defect types


# Paths
training_folder_path = Path(f"mvtec/{SUBJECT}/train/good")
testing_folder_path = Path(f"mvtec/{SUBJECT}/test")

# Transformations
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Load backbone model
backbone = ResNetFeatureExtractor().to(config.DEVICE)

# Create memory bank
memory_bank = create_memory_bank(backbone, training_folder_path, transform)

# Calculate scores for training set
y_score_training = get_y_score_from_training_set(
    training_folder_path, transform, memory_bank, backbone
)

best_threshold = np.mean(y_score_training) + 2 * np.std(y_score_training)

plt.hist(y_score_training, bins=50)
plt.vlines(x=best_threshold, ymin=0, ymax=30, color="r")
plt.xlabel("Anomaly Score")
plt.ylabel("Frequency")
plt.show()

# Calculate scores for test set
y_score, y_true = get_y_score_and_true_from_testing(
    transform, backbone, memory_bank, testing_folder_path
)


# ROC and F1 evaluation
auc_roc_score, fpr, tpr, thresholds = calculate_roc_auc(y_true, y_score)

plot_roc_curve(fpr, tpr, auc_roc_score)

f1_scores = [f1_score(y_true, y_score >= threshold) for threshold in thresholds]
best_threshold = thresholds[np.argmax(f1_scores)]
ic(best_threshold)
plot_confusion_matrix(y_true, (y_score >= best_threshold).astype(int))


# Prediction on the test set
testing_defect_folder_path = Path(f"mvtec/{SUBJECT}/test/{DEFECT_TYPE}")
predict_on_test_set(
    testing_defect_folder_path, transform, backbone, memory_bank, best_threshold
)
