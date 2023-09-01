import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, has_file_allowed_extension
from tqdm import tqdm


class SelectedImageFolder(Dataset):
    def __init__(self, root, selected_folders, transform=None):
        super(Dataset, self).__init__()

        # Initialization
        self.root = root
        self.transform = transform
        self.extensions = IMG_EXTENSIONS

        # Create a list of all valid images in the selected folders
        self.samples = []

        for idx, folder in enumerate(tqdm(selected_folders)):
            folder_path = os.path.join(root, folder)
            if not os.path.isdir(folder_path):
                continue
            for rootdir, _, filenames in os.walk(folder_path):
                for filename in filenames:
                    if has_file_allowed_extension(filename, self.extensions):
                        path = os.path.join(rootdir, filename)
                        self.samples.append((path, idx))

        # Define the classes based on the selected folders
        self.classes = selected_folders
        self.class_to_idx = {folder: idx for idx, folder in enumerate(self.classes)}

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target

    def __len__(self):
        return len(self.samples)
