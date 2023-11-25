"""Models implementation"""
from typing import Dict, Optional

from torch import Tensor, nn
from opr.modules import Concat


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


class LateFusionModel(nn.Module):
    """Meta-model for multimodal Place Recognition architectures with late fusion."""

    def __init__(
        self,
        image_module: Optional[nn.Module] = None,
        semantic_module: Optional[nn.Module] = None,
        cloud_module: Optional[nn.Module] = None,
        text_module: Optional[nn.Module] = None,
        fusion_module: Optional[nn.Module] = None,
    ) -> None:
        """Meta-model for multimodal Place Recognition architectures with late fusion.

        Args:
            image_module (nn.Module, optional): Image modality branch. Defaults to None.
            semantic_module (nn.Module, optional): Semantic modality branch. Defaults to None.
            cloud_module (nn.Module, optional): Cloud modality branch. Defaults to None.
            text_module (nn.Module, optional): Text modality branch. Defaults to None.
            fusion_module (nn.Module, optional): Module to fuse different modalities.
                If None, will be set to opr.modules.Concat(). Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.semantic_module = semantic_module
        self.cloud_module = cloud_module
        self.text_module = text_module
        if fusion_module:
            self.fusion_module = fusion_module
        else:
            self.fusion_module = Concat()

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:  # noqa: D102
        out_dict: Dict[str, Tensor] = {}

        if self.image_module is not None:
            out_dict["image"] = self.image_module(batch)["final_descriptor"]

        if self.semantic_module is not None:
            out_dict["semantic"] = self.semantic_module(batch)["final_descriptor"]

        if self.cloud_module is not None:
            out_dict["cloud"] = self.cloud_module(batch)["final_descriptor"]

        if self.text_module is not None:
            out_dict["text"] = self.text_module(batch)["final_descriptor"]

        out_dict["final_descriptor"] = self.fusion_module(out_dict)

        return out_dict
