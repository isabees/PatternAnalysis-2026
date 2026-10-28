# Contains source code for training, validating, testing, and saving model. model imported from modules.py, data loader imported from dataset.py. plot losses & metrics during training
import time
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import modules
import dataset

MODEL_NAME = "BasicCNN"
# MODEL_NAME = "ConvNeXt"

EPOCHS = 30
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0001
NUM_CLASSES = 2
LABEL_CLASSES = {"AD": 0, "CN": 1}
class_names = ["AD", "CN"]

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if MODEL_NAME == "BasicCNN":
    CHECKPOINT_PATH = OUTPUT_DIR / "basic_cnn.pth"
    METRICS_PATH = OUTPUT_DIR / "metrics.json"
    HISTORY_PATH = OUTPUT_DIR / "training_history.json"
else: # MODEL_NAME = ConvNeXt
    CHECKPOINT_PATH = OUTPUT_DIR / "convnext.pth"
    METRICS_PATH = OUTPUT_DIR / "convnext_metrics.json"
    HISTORY_PATH = OUTPUT_DIR / "convnext_history.json"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# initialise model
if MODEL_NAME == "BasicCNN":
    model = modules.BasicCNN(num_classes=NUM_CLASSES).to(device)
else:
    model = modules.ConvNeXt(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

parameter_count = sum(p.numel() for p in model.parameters())
trainable_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Device:", device)
print("Total parameters:", parameter_count)
print("Trainable parameters:", trainable_parameter_count)

# training
def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()

        optimiser.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return total_loss / total, correct / total

# evaluation
def evaluate(model, loader, criterion, device):
    """Evaluate loss, accuracy, predictions and probabilities."""

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()

            outputs = model(images)

            loss = criterion(outputs, labels)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            correct += (predictions == labels).sum().item()
            total += batch_size

            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    probabilities_np = np.asarray(all_probabilities)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "labels": np.asarray(all_labels, dtype=int),
        "predictions": np.asarray(all_predictions, dtype=int),
        "probabilities": probabilities_np,
        "total": total,
    }

def synchronise_cuda():
    """Ensure GPU operations finish before timing or reading memory."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# metric stuff
history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "train_epoch_seconds": [],
    "val_epoch_seconds": [],
    "epoch_seconds": [],
}

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

total_start = time.perf_counter()

# training and validation
for epoch in range(EPOCHS):
    epoch_start = time.perf_counter()

    # training
    synchronise_cuda()
    train_start = time.perf_counter()

    train_loss, train_acc = train_one_epoch(
        model,
        dataset.train_loader,
        criterion,
        optimiser,
        device,
    )

    synchronise_cuda()
    train_seconds = time.perf_counter() - train_start

    # validation
    synchronise_cuda()
    val_start = time.perf_counter()

    val_results = evaluate(
        model,
        dataset.val_loader,
        criterion,
        device,
    )

    synchronise_cuda()
    val_seconds = time.perf_counter() - val_start
    epoch_seconds = time.perf_counter() - epoch_start

    # store metrics
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_results["loss"])
    history["train_accuracy"].append(train_acc)
    history["val_accuracy"].append(val_results["accuracy"])
    history["train_epoch_seconds"].append(train_seconds)
    history["val_epoch_seconds"].append(val_seconds)
    history["epoch_seconds"].append(epoch_seconds)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_results['loss']:.4f} | "
        f"Val Acc: {val_results['accuracy']:.4f} | "
        f"Train Time: {train_seconds:.2f}s | "
        f"Val Time: {val_seconds:.2f}s"
    )

synchronise_cuda()
training_seconds = time.perf_counter() - total_start

# test evaluation
synchronise_cuda()
test_start = time.perf_counter()

test_results = evaluate(
    model,
    dataset.test_loader,
    criterion,
    device,
)

synchronise_cuda()
test_seconds = time.perf_counter() - test_start

print("\nFinal test results")
print(f"Test Loss: {test_results['loss']:.4f}")
print(f"Test Accuracy: {test_results['accuracy']:.4f}")
print(f"Test Runtime: {test_seconds:.2f} seconds")

y_true = test_results["labels"]
y_pred = test_results["predictions"]
probabilities = test_results["probabilities"]

# confusion matrix: rows = actual, columns = predicted.
confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)

for actual, predicted in zip(y_true, y_pred):
    confusion[actual, predicted] += 1

per_class_metrics = {}

for i, class_name in enumerate(class_names):
    tp = int(confusion[i, i])
    fp = int(confusion[:, i].sum() - tp)
    fn = int(confusion[i, :].sum() - tp)
    support = int(confusion[i, :].sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    per_class_metrics[class_name] = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "support": support,
    }

    print(
        f"{class_name}: "
        f"Precision={precision:.4f}, "
        f"Recall={recall:.4f}, "
        f"F1={f1:.4f}, "
        f"Support={support}"
    )

print("\nConfusion matrix (actual rows, predicted columns):")
print(confusion)

# GPU metrics
if device.type == "cuda":
    peak_allocated_mib = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    )
    peak_reserved_mib = (
        torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    )
else:
    peak_allocated_mib = None
    peak_reserved_mib = None

print(f"\nTraining runtime: {training_seconds:.2f} seconds")
print(f"Test runtime: {test_seconds:.2f} seconds")

if device.type == "cuda":
    print(f"Peak allocated GPU VRAM: {peak_allocated_mib:.2f} MiB")
    print(f"Peak reserved GPU VRAM: {peak_reserved_mib:.2f} MiB")
else:
    print("Peak GPU VRAM: N/A (CPU execution)")

# summary
print("\nAll training and evaluation tasks completed.")
print(f"Training time: {training_seconds:.2f} seconds")
print(f"Test time: {test_seconds:.2f} seconds")
print(f"Parameters: {parameter_count}")
print(f"Test loss: {test_results['loss']:.4f}")
print(f"Test accuracy: {test_results['accuracy']:.4f}")
print(f"Results directory: {OUTPUT_DIR.resolve()}")

# save baseline model
checkpoint = {
    "model_name": MODEL_NAME,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimiser.state_dict(),
    "num_classes": NUM_CLASSES,
    "class_names": class_names,
    "epoch": EPOCHS,
    "parameter_count": parameter_count,
    "trainable_parameter_count": trainable_parameter_count,
    "history": history,
    "test_loss": test_results["loss"],
    "test_accuracy": test_results["accuracy"],
    "training_seconds": training_seconds,
    "test_seconds": test_seconds,
    "peak_gpu_allocated_mib": peak_allocated_mib,
    "peak_gpu_reserved_mib": peak_reserved_mib,
}
torch.save(checkpoint, CHECKPOINT_PATH)
print(f"\nModel checkpoint saved to: {CHECKPOINT_PATH}")

metrics = {
    "model_name": "MODEL_NAME",
    "device": str(device),
    "epochs": EPOCHS,
    "parameter_count": parameter_count,
    "trainable_parameter_count": trainable_parameter_count,
    "class_names": class_names,
    "training_seconds": training_seconds,
    "test_seconds": test_seconds,
    "total_train_seconds": sum(history["train_epoch_seconds"]),
    "total_validation_seconds": sum(history["val_epoch_seconds"]),
    "peak_gpu_allocated_mib": peak_allocated_mib,
    "peak_gpu_reserved_mib": peak_reserved_mib,
    "test_loss": test_results["loss"],
    "test_accuracy": test_results["accuracy"],
    "test_samples": test_results["total"],
    "test_confusion_matrix": confusion.tolist(),
    "test_per_class_metrics": per_class_metrics,
}

with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

with open(HISTORY_PATH, "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)

print(f"Metrics saved to: {METRICS_PATH}")
print(f"Training history saved to: {HISTORY_PATH}")
