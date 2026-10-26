# Shows example usage of trained model. load saved model weights, run inference on test cases, print out results, and provide visualisations where applicable
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import modules
import dataset

OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / "basic_cnn.pth"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the saved baseline model
checkpoint = torch.load(
    CHECKPOINT_PATH, map_location=device, weights_only=True
)

num_classes = checkpoint["num_classes"]
class_names = checkpoint["class_names"]
history = checkpoint["history"]

model = modules.BasicCNN(num_classes=num_classes).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Device:", device)
print("\nCheckpoint loaded successfully.")
print("Model:", checkpoint["model_name"])
print("Epochs:", checkpoint["epoch"])
print("Class names:", class_names)

# load saved metrics and history
metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

if HISTORY_PATH.exists():
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        history = json.load(f)

# display saved performance and resource metrics
print("\n~ SAVED MODEL RESULTS ~")
for key in [
    "training_seconds",
    "test_seconds",
    "test_loss",
    "test_accuracy",
    "parameter_count",
    "trainable_parameter_count",
    "peak_gpu_allocated_mib",
    "peak_gpu_reserved_mib",
]:
    value = checkpoint.get(key)
    print(f"{key}: {value}")

# training and validation loss
epochs = range(1, len(history["train_loss"]) + 1)

for metric, ylabel, title, filename in [
    ("loss", "Cross-Entropy Loss", "Training and Validation Loss",
     "loss_curves.png"),
    ("accuracy", "Accuracy", "Training and Validation Accuracy",
     "accuracy_curves.png"),
]:
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history[f"train_{metric}"], label="Training")
    plt.plot(epochs, history[f"val_{metric}"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    if metric == "accuracy":
        plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

# confusion matrix
confusion = metrics.get("test_confusion_matrix")

if confusion is not None:
    confusion = np.asarray(confusion, dtype=int)

    plt.figure(figsize=(5, 4))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar(label="Number of samples")
    plt.xticks(range(num_classes), class_names)
    plt.yticks(range(num_classes), class_names)
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Test Confusion Matrix")

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(confusion[i, j]),
                     ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_confusion_matrix.png", dpi=300)
    plt.close()
else:
    print("No confusion matrix found in metrics.json.")

# run inference to get test probabilities
all_probabilities = []

with torch.no_grad():
    for images, _ in dataset.test_loader:
        images = images.to(device)
        probabilities = torch.softmax(model(images), dim=1)
        all_probabilities.extend(probabilities.cpu().tolist())

probabilities = np.asarray(all_probabilities)

# test probability distributions
fig, axes = plt.subplots(1, num_classes, figsize=(10, 4), squeeze=False)

for i, class_name in enumerate(class_names):
    axes[0, i].hist(probabilities[:, i], bins=20, range=(0, 1),
                    edgecolor="black")
    axes[0, i].set_title(f"Predicted P({class_name})")
    axes[0, i].set_xlabel("Softmax Probability")
    axes[0, i].set_ylabel("Number of Samples")

fig.suptitle("Test Set Class Probability Distributions")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "test_probability_distributions.png", dpi=300)
plt.close(fig)

print("\nGenerated plots:")
print(" - loss_curves.png")
print(" - accuracy_curves.png")
print(" - test_confusion_matrix.png")
print(" - test_probability_distributions.png")