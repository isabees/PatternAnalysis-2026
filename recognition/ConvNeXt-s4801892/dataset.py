# Contains data loader for loading & preprocessing data, including leakagefree train/validation/test splitting functions and data augmentations
from pathlib import Path
from sklearn.model_selection import train_test_split

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

root = Path("/home/groups/comp3710/ADNI/AD_NC")

images = [
    (f, label, f.stem.split("_")[0])
    for label, cls in [(1, "AD"), (0, "NC")] # label: AD = 1, NC = 0
    for f in (root / "train" / cls).glob("*.jpeg")
]

# print("Dataset root:", root.resolve())
# print("Root exists:", root.exists())
# print("Train AD exists:", (root / "train" / "AD").exists())
# print("Train NC exists:", (root / "train" / "NC").exists())
# print("AD JPEGs:", len(list((root / "train" / "AD").glob("*.jpeg"))))
# print("NC JPEGs:", len(list((root / "train" / "NC").glob("*.jpeg"))))

# split unique patients, stratified by diagnosis
patients = list({(pid, label) for _, label, pid in images})
patient_ids, labels = zip(*patients)

train_ids, val_ids = train_test_split(
    patient_ids,
    test_size=15/85,
    random_state=42,
    stratify=labels,
)

train_ids, val_ids = set(train_ids), set(val_ids)

train_images = [(f, y) for f, y, pid in images if pid in train_ids]
val_images = [(f, y) for f, y, pid in images if pid in val_ids]

# Keep the original test set unchanged
test_images = [
    (f, label)
    for label, cls in [(1, "AD"), (0, "NC")]
    for f in (root / "test" / cls).glob("*.jpeg")
]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1156], std=[0.2229])
])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1156], std=[0.2229])
])  

class MRIDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]

        image = Image.open(path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(
    MRIDataset(train_images, train_transform),
    batch_size=32,
    shuffle=True,
)

val_loader = DataLoader(
    MRIDataset(val_images, transform),
    batch_size=32,
    shuffle=False,
)

test_loader = DataLoader(
    MRIDataset(test_images, transform),
    batch_size=32,
    shuffle=False,
)

images, labels = next(iter(train_loader))

# print(images.shape)  # [32, 1, 224, 224]
# print(labels.shape)  # [32]
# print(labels[:5])    # 0 = NC, 1 = AD

# verifying the data split (make sure no patient appears in multiple splits):
# def patient_ids(data):
#     return {path.stem.split("_")[0] for path, label in data}

# train_ids = patient_ids(train_images)
# val_ids = patient_ids(val_images)
# test_ids = patient_ids(test_images)

# assert not (train_ids & val_ids), "Leakage: train and validation!"
# assert not (train_ids & test_ids), "Leakage: train and test!"
# assert not (val_ids & test_ids), "Leakage: validation and test!"

# print("No patient leakage detected.")

# for split_name, data in [
#     ("train", train_images),
#     ("validation", val_images),
#     ("test", test_images),
# ]:
#     for path, label in data:
#         assert label in (0, 1), f"Invalid label: {path}"
#         assert path.parent.name == ("AD" if label == 1 else "NC"), \
#             f"Incorrect label: {path}"

#         with Image.open(path) as img:
#             img.verify()

#     print(f"{split_name}: {len(data)} images verified.")

# print("All checks passed!")