import multiprocessing
from joblib import Parallel, delayed
from torch.utils.data import Dataset
import sys
from PIL import Image


def load_img(path):
    return Image.open(path).convert("RGB")


def estimate_image_size(img):
    """
    Estima memória de uma imagem PIL ou array NumPy em bytes.
    """
    try:
        import numpy as np
        if hasattr(img, "size") and hasattr(img, "mode"):  # PIL
            # RGB = 3 canais, L = 1 canal, etc
            channels = len(img.getbands())
            w, h = img.size
            return w * h * channels * 4  # assumindo float32 após transform
        elif isinstance(img, np.ndarray):
            return img.nbytes
        else:
            return sys.getsizeof(img)
    except:
        return sys.getsizeof(img)

class ChestXrayDataset(Dataset):
    def __init__(self, data, transforms=None, cache=False, max_cache_gb=None, n_jobs=-1):
        """
        data: ImageFolder
        transforms: torchvision transforms
        cache: True para pré-carregar imagens
        max_cache_gb: limite de memória para caching (GB)
        """
        self._data = data
        self._transforms = transforms
        self._cache = cache
        self._cached_imgs = {}

        if cache:
            num_workers = multiprocessing.cpu_count() if n_jobs == -1 else n_jobs
            max_cache_bytes = max_cache_gb * 1024**3 if max_cache_gb else None
            current_cache_bytes = 0

            def load_and_return(idx):
                path, _ = self._data.samples[idx]
                img = load_img(path)
                size = estimate_image_size(img)
                return idx, img, size

            results = Parallel(n_jobs=num_workers)(
                delayed(load_and_return)(idx) for idx in range(len(self._data))
            )

            # Adiciona imagens ao cache até atingir limite
            for idx, img, size in results:
                if max_cache_bytes is None or current_cache_bytes + size <= max_cache_bytes:
                    self._cached_imgs[idx] = img
                    current_cache_bytes += size
                else:
                    break  # não adiciona mais imagens

            print(f"Cache carregado: {len(self._cached_imgs)}/{len(self._data)} imagens, "
                  f"~= {current_cache_bytes / 1024**2:.2f} MB")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        path, label = self._data.samples[idx]
        if self._cache and idx in self._cached_imgs:
            img = self._cached_imgs[idx]
        else:
            img = load_img(path)

        if self._transforms:
            img = self._transforms(img)

        return img, label
