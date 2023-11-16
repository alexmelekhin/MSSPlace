"""Models implementation"""
from typing import Dict, Optional

from torch import Tensor, nn


class TextModel(nn.Module):
    """Meta-model for text-based Place Recognition."""

    def __init__(
        self,
        model: nn.Module,
        fusion: Optional[nn.Module] = None,
    ) -> None:
        """Meta-model for text-based Place Recognition.

        Args:
            model (nn.Module): Text backbone.
            fusion (nn.Module, optional): Module to fuse descriptors for multiple texts in batch.
                Defaults to None.
        """
        super().__init__()
        self.model = model
        self.fusion = fusion

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        text_descriptors = {}
        for key, value in batch.items():
            if key.startswith("texts_"):
                text_descriptors[key] = self.model(value)
        if len(text_descriptors) > 1:
            if self.fusion is None:
                raise ValueError("Fusion module is not defined but multiple texts are provided")
            descriptor = self.fusion(text_descriptors)
        else:
            if self.fusion is not None:
                raise ValueError("Fusion module is defined but only one text is provided")
            descriptor = list(text_descriptors.values())[0]
        out_dict: Dict[str, Tensor] = {"final_descriptor": descriptor}
        return out_dict
