import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

class MelanomaDataset(Dataset):
    LABEL_MAP = {
        'BCC': 0, 'SCC': 1, 'ACK': 2, 'SEK': 3, 
        'BOD': 4, 'MEL': 5, 'NEV': 6
    }
    NUM_CLASSES = 7 # Added for clarity

    def __init__(self, csv_path, base_img_dir, split, transform=None, 
                 target_col='diagnostic', img_col='img_id', ml_set_col='ml_set'):
        """
        Args:
            csv_path (string): Path to the preprocessed csv file with 'ml_set' column.
            base_img_dir (string): Base directory containing 'train', 'val', 'test' subfolders.
            split (string): Which split to load ('train', 'val', or 'test').
            transform (callable, optional): Optional transform to be applied on a sample.
            target_col (string): Name of the column containing the target labels.
            img_col (string): Name of the column containing the image identifiers.
            ml_set_col (string): Name of the column indicating the dataset split ('train', 'val', 'test').
        """
        if split not in ['train', 'val', 'test']:
            raise ValueError("split must be one of 'train', 'val', or 'test'")

        try:
            full_metadata = pd.read_csv(csv_path)
        except FileNotFoundError:
             raise FileNotFoundError(f"Metadata CSV not found at {csv_path}")
        except Exception as e:
            raise IOError(f"Error reading metadata CSV {csv_path}: {e}")
            
        if ml_set_col not in full_metadata.columns:
            raise KeyError(f"Split column '{ml_set_col}' not found in {csv_path}. Ensure preprocessing was run.")

        # Filter metadata for the specified split
        self.metadata = full_metadata[full_metadata[ml_set_col] == split].reset_index(drop=True)
        
        if self.metadata.empty:
            print(f"Warning: No samples found for split '{split}' in {csv_path}")

        self.base_img_dir = base_img_dir
        self.split = split # Store the split name ('train', 'val', 'test')
        self.transform = transform
        self.target_col = target_col
        self.img_col = img_col

        # Validate target_col contains expected labels if it's the first item loaded
        if not self.metadata.empty:
            first_label = self.metadata.loc[0, self.target_col]
            if first_label not in self.LABEL_MAP:
                 print(f"Warning: First label '{first_label}' in split '{split}' is not in the expected LABEL_MAP: {list(self.LABEL_MAP.keys())}. Ensure target_col is correct.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image ID and label from the filtered metadata for the current split
        img_id = self.metadata.loc[idx, self.img_col]
        label_str = self.metadata.loc[idx, self.target_col]

        try:
            label_idx = self.LABEL_MAP[label_str]
            # Create one-hot encoded label
            label = torch.zeros(self.NUM_CLASSES, dtype=torch.float32)
            label[label_idx] = 1.0
        except KeyError:
            print(f"Error: Unexpected label '{label_str}' found for image {img_id} in split '{self.split}'. Expected one of {list(self.LABEL_MAP.keys())}.")
            # Return None for both image and label to be filtered by collate_fn
            return None, None

        # Construct path: base_img_dir / split_name / image_id.ext
        img_filename = f"{str(img_id)}"
        img_path = os.path.join(self.base_img_dir, self.split, img_filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found for split '{self.split}' at {img_path}")
            # Return None values, handled by collate_fn
            # Ensure label is also None if image fails
            return None, None 
        except Exception as e:
            print(f"Error opening image {img_path}: {e}")
             # Ensure label is also None if image fails
            return None, None

        if self.transform:
            image = self.transform(image)

        return image, label

def get_default_transforms(img_size=224):
    # Default transforms using ImageNet mean and std
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_dataloader(csv_path, base_img_dir, split, batch_size=32, num_workers=4, img_size=224, pin_memory=True, transform=None, **kwargs):
    """
    Creates a DataLoader for the MelanomaDataset based on the specified split.

    Args:
        csv_path (string): Path to the preprocessed CSV file (containing 'ml_set' column).
        base_img_dir (string): Base directory containing 'train', 'val', 'test' image subfolders.
        split (string): Which split to load ('train', 'val', or 'test'). Determines shuffling.
        batch_size (int): How many samples per batch to load.
        num_workers (int): How many subprocesses to use for data loading.
        img_size (int): The target size for image resizing (used for default transforms).
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory.
        transform (callable, optional): Custom transform pipeline. If None, uses default transforms.

    Returns:
        torch.utils.data.DataLoader: DataLoader instance for the specified split.
    """
    if transform is None:
        transform = get_default_transforms(img_size)

    dataset = MelanomaDataset(csv_path=csv_path, 
                              base_img_dir=base_img_dir, 
                              split=split, 
                              transform=transform)

    # Determine shuffle based on split
    shuffle = (split == 'train') 
    print(f"Creating DataLoader for split: '{split}', Shuffle: {shuffle}")


    # Handle potential None returns from __getitem__ if files are missing or corrupted
    def collate_fn(batch):
        # Filter out samples where image loading failed (returned None)
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch: # If all items in batch were None or the dataset split is empty
             return torch.tensor([]), torch.tensor([]) # Return empty tensors
        # Use default collate for the valid samples
        return torch.utils.data.dataloader.default_collate(batch)


    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle, # Set based on split
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=collate_fn) # Use custom collate_fn
    return dataloader
