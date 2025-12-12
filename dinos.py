import os
from typing import List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class DINOS(Dataset):
    H: int = 256
    W: int = 256
    C: int = 3

    def __init__(
        self,
        folder: str,
        extensions: List[str] = [".png", ".jpg", ".jpeg"],
        recursive: bool = False,
        return_paths: bool = False,
        transform=None,
    ):
        self.folder = folder
        self.return_paths = return_paths
        self.transform = transform

        self.files = self._collect_files(folder, extensions, recursive)
        if not self.files:
            raise ValueError(f"No image files found in {folder}")

    def _collect_files(self, folder: str, extensions: List[str], recursive: bool) -> List[str]:
        exts = tuple(ext.lower() for ext in extensions)
        if not recursive:
            return sorted(
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(exts)
            )
        out = []
        for root, _, filenames in os.walk(folder):
            for f in filenames:
                if f.lower().endswith(exts):
                    out.append(os.path.join(root, f))
        return sorted(out)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        img = img.resize((self.W, self.H))

        if self.transform is not None:
            x = self.transform(img)  # usually returns a torch.Tensor
        else:
            # Convert to torch tensor in CHW format
            arr = np.asarray(img, dtype=np.float32) / 255.0
            x = torch.from_numpy(arr).permute(0, 1, 2)  # CHW

        if self.return_paths:
            return x, path
        return x

