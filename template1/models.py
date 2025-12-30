import torch.nn as nn
from abc import ABC, abstractmethod
from .config import BaseConfig


class BaseModel(nn.Module, ABC):
    """Abstract base model class"""
    
    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, x):
        """Forward pass - implement in subclass"""
        pass
    
    def get_num_params(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)