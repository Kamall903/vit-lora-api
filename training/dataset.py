
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor
from sklearn.model_selection import train_test_split

from training.config import (
    MODEL_NAME,
    DATA_DIR,
    BATCH_SIZE,
    SEED
)


class EuroSATDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]

        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


def get_dataloaders():
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    dataset = ImageFolder(DATA_DIR)

    indices = list(range(len(dataset)))
    labels = dataset.targets

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=SEED,
        stratify=labels
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=SEED,
        stratify=[labels[i] for i in temp_idx]
    )

    def make_loader(indices, shuffle):
        subset = Subset(dataset, indices)
        processed = EuroSATDataset(subset, processor)

        return DataLoader(
            processed,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )

    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)

    return (
        train_loader,
        val_loader,
        test_loader,
        dataset.classes
    )


if __name__ == "__main__":
    train_loader, val_loader, test_loader, classes = (
        get_dataloaders()
    )

    print("Classes:", classes)
    print("Training images:", len(train_loader.dataset))
    print("Validation images:", len(val_loader.dataset))
    print("Testing images:", len(test_loader.dataset))

    batch = next(iter(train_loader))

    print("Image batch:", batch["pixel_values"].shape)
    print("Labels:", batch["labels"].shape)