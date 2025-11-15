import torch
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, path):
        dataset_metadata_dict = torch.load(path)
        self.text = dataset_metadata_dict["text"]
        self.img = dataset_metadata_dict["img"]
        self.text_mask = dataset_metadata_dict.get("text_mask", torch.ones(self.text.shape[0], self.text.shape[1], dtype=torch.long))
        self.img_mask = dataset_metadata_dict.get("img_mask", torch.ones(self.img.shape[0], self.img.shape[1], dtype=torch.long))
        self.labels = dataset_metadata_dict["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text_embedding": self.text[idx],
            "image_embedding": self.img[idx],
            "text_mask": self.text_mask[idx],
            "image_mask": self.img_mask[idx],
            "label": self.labels[idx]
        }