import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os

class MelanomaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, target_col='target', img_col='image_id', file_ext='.jpg'):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_col (string): Name of the column containing the target labels.
            img_col (string): Name of the column containing the image identifiers.
            file_ext (string): File extension of the images (e.g., '.jpg', '.png').
        """
        self.metadata = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_col = target_col
        self.img_col = img_col
        self.file_ext = file_ext

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir,
                                self.metadata.loc[idx, self.img_col] + self.file_ext)
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image file not found at {img_name}")
            # Return a placeholder or raise an error, depending on desired behavior
            # For simplicity, returning None here. Handle appropriately in collation.
            # Or better, ensure CSV/img_dir are correct before starting.
            return None, None # Or raise specific error


        label = int(self.metadata.loc[idx, self.target_col])

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

def get_dataloader(csv_path, img_dir, batch_size=32, shuffle=True, num_workers=4, img_size=224, pin_memory=True, transform=None):
    """
    Creates a DataLoader for the custom dataset.

    Args:
        csv_path (string): Path to the csv file.
        img_dir (string): Directory with images.
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Set to True to have the data reshuffled at every epoch.
        num_workers (int): How many subprocesses to use for data loading.
        img_size (int): The target size for image resizing.
        pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        transform (callable, optional): Custom transform pipeline. If None, uses default transforms.

    Returns:
        torch.utils.data.DataLoader: DataLoader instance.
    """
    if transform is None:
        transform = get_default_transforms(img_size)

    dataset = MelanomaDataset(csv_file=csv_path, img_dir=img_dir, transform=transform)

    # Handle potential None returns from __getitem__ if files are missing
    def collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if not batch: # If all items in batch were None
             return torch.tensor([]), torch.tensor([])
        return torch.utils.data.dataloader.default_collate(batch)


    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            collate_fn=collate_fn) # Use custom collate_fn
    return dataloader

# Example usage (optional):
# if __name__ == '__main__':
#     # Create dummy csv and image folders/files for testing if needed
#     # For example:
#     # os.makedirs('dummy_images', exist_ok=True)
#     # Image.new('RGB', (60, 30), color = 'red').save('dummy_images/img1.jpg')
#     # Image.new('RGB', (60, 30), color = 'blue').save('dummy_images/img2.jpg')
#     # df = pd.DataFrame({'image_id': ['img1', 'img2'], 'target': [0, 1]})
#     # df.to_csv('dummy_metadata.csv', index=False)

#     # train_loader = get_dataloader('dummy_metadata.csv', 'dummy_images', batch_size=1)
#     # for images, labels in train_loader:
#     #     print(images.shape, labels.shape)
#     #     # Clean up dummy files/folders afterwards if created
#     #     # os.remove('dummy_metadata.csv')
#     #     # shutil.rmtree('dummy_images')
#     pass 