"""
Data handling classes
"""
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from .config import BaseConfig


class BaseDataset(Dataset, ABC):
    """
    Abstract base dataset
    """
    
    def __init__(self, data_path: Path, transform: Optional[Callable] = None):
        self.data_path = data_path
        self.transform = transform
        self.data = self._load_data()
    
    @abstractmethod
    def _load_data(self):
        """Load data from disk - implement in subclass"""
        pass
    
    @abstractmethod
    def __len__(self):
        """Return dataset size"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int):
        """Get item by index"""
        pass


class DataModule:
    """Handles data loading and preprocessing"""
    
    def __init__(self, 
                 config: BaseConfig,
                 train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None):
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )
    
    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == 'cuda' else False
        )
    
    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )