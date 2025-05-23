from typing import Optional

import MinkowskiEngine as ME  # noqa: N817
import torch
from torch import nn, Tensor
from opr.models.place_recognition.base import ImageModel, SemanticModel, CloudModel
from opr.modules import Concat

_modalities = ("image", "cloud", "semantic", "text")


class LateFusionModel(nn.Module):
    """Meta-model for multimodal Place Recognition architectures with late fusion."""

    def __init__(
        self,
        image_module: Optional[ImageModel] = None,
        semantic_module: Optional[SemanticModel] = None,
        cloud_module: Optional[CloudModel] = None,
        text_module: Optional[nn.Module] = None,
        soc_module: Optional[nn.Module] = None,
        fusion_module: Optional[nn.Module] = None,
    ) -> None:
        """Meta-model for multimodal Place Recognition architectures with late fusion.

        Args:
            image_module (ImageModule, optional): Image modality branch. Defaults to None.
            semantic_module (SemanticModel, optional): Semantic modality branch. Defaults to None.
            cloud_module (CloudModule, optional): Cloud modality branch. Defaults to None.
            soc_module (nn.Module, optional): Module to fuse different modalities.
            fusion_module (FusionModule, optional): Module to fuse different modalities.
                If None, will be set to opr.modules.Concat(). Defaults to None.
        """
        super().__init__()

        self.image_module = image_module
        self.semantic_module = semantic_module
        self.cloud_module = cloud_module
        self.text_module = text_module
        self.soc_module = soc_module
        if fusion_module:
            self.fusion_module = fusion_module
        else:
            self.fusion_module = Concat()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: D102
        out_dict: dict[str, Tensor] = {}

        if self.image_module is not None:
            out_dict["image"] = self.image_module(batch)["final_descriptor"]

        if self.semantic_module is not None:
            out_dict["semantic"] = self.semantic_module(batch)["final_descriptor"]

        if self.cloud_module is not None:
            out_dict["cloud"] = self.cloud_module(batch)["final_descriptor"]

        if self.text_module is not None:
            out_dict["text"] = self.text_module(batch)["final_descriptor"]

        if self.soc_module is not None:
            out_dict["soc"] = self.soc_module(batch)["final_descriptor"]

        out_dict = self.fusion_module(out_dict)

        if not isinstance(out_dict, dict):
            out_dict = {"final_descriptor": out_dict}

        return out_dict


class MiddleFusionModel(LateFusionModel):
    def __init__(
        self,
        image_module: Optional[ImageModel] = None,
        semantic_module: Optional[SemanticModel] = None,
        cloud_module: Optional[CloudModel] = None,
        soc_module: Optional[nn.Module] = None,
        fusion_module: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(image_module, semantic_module, cloud_module, soc_module, fusion_module)
        self.cloud_dim_reduction = ME.MinkowskiAvgPooling(kernel_size=3, stride=3, dimension=3)
        self.final_fusion = Concat()

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:  # noqa: D102
        ### step 1: feature extraction
        if self.image_module is not None:
            img_features = {}
            img_features_shapes = {}
            for key, value in batch.items():
                if key.startswith("images_"):
                    img_features[key] = self.image_module.backbone(value)
                    img_features_shapes[key] = img_features[key].shape
                    img_features[key] = (
                        img_features[key]
                        .view(img_features[key].shape[0], img_features[key].shape[1], -1)
                        .permute(0, 2, 1)
                    )  # (B, N_feats, Desc_dim)
        if self.semantic_module is not None:
            semantic_features = {}
            semantic_features_shapes = {}
            for key, value in batch.items():
                if key.startswith("masks_"):
                    semantic_features[key] = self.semantic_module.backbone(value)
                    semantic_features_shapes[key] = semantic_features[key].shape
                    semantic_features[key] = (
                        semantic_features[key]
                        .view(semantic_features[key].shape[0], semantic_features[key].shape[1], -1)
                        .permute(0, 2, 1)
                    )  # (B, N_feats, Desc_dim)
        if self.cloud_module is not None:
            sparse_voxel = ME.SparseTensor(
                features=batch["pointclouds_lidar_feats"], coordinates=batch["pointclouds_lidar_coords"]
            )
            sparse_cloud_features = self.cloud_module.backbone(sparse_voxel)
            sparse_cloud_features = self.cloud_dim_reduction(sparse_cloud_features)
        # TODO: add text model

        ### step 2: transformer interaction
        tokens_dict = {}
        if self.image_module is not None:
            tokens_dict["image"] = torch.cat(list(img_features.values()), dim=1)
        if self.semantic_module is not None:
            tokens_dict["semantic"] = torch.cat(list(semantic_features.values()), dim=1)
        if self.cloud_module is not None:
            min_coordinate = torch.tensor(
                [
                    torch.min(sparse_cloud_features.C[:, 1]),
                    torch.min(sparse_cloud_features.C[:, 2]),
                    torch.min(sparse_cloud_features.C[:, 3]),
                ]
            )
            dense_cloud_features, _, _ = sparse_cloud_features.dense(min_coordinate=min_coordinate)
            dense_cloud_shape = dense_cloud_features.shape
            dense_cloud_features = dense_cloud_features.view(
                dense_cloud_features.shape[0], dense_cloud_features.shape[1], -1
            ).permute(0, 2, 1)  # (B, N_feats, Desc_dim)
            tokens_dict["cloud"] = dense_cloud_features
        tokens_dict = self.fusion_module(tokens_dict)

        ### step 3: back into initial states and finish processing
        out_dict = {}
        if self.image_module is not None:
            image_feat_lens = [s[-1] * s[-2] for s in img_features_shapes.values()]
            img_features_list = torch.split(tokens_dict["image"], image_feat_lens, dim=1)
            for key, feats in zip(list(img_features.keys()), img_features_list):
                img_features[key] = feats.permute(0, 2, 1).view(*img_features_shapes[key])
                img_features[key] = self.image_module.head(img_features[key])
            out_dict["image"] = self.image_module.fusion(img_features)
        if self.cloud_module is not None:
            dense_cloud_features = tokens_dict["cloud"].permute(0, 2, 1).view(*dense_cloud_shape)
            out_dict["cloud"] = self.cloud_module.head(ME.to_sparse(dense_cloud_features))
        out_dict["final_descriptor"] = self.final_fusion(out_dict)
        return out_dict


class TransformerModalityInteraction(nn.Module):
    def __init__(
        self,
        desc_dim: int = 256,
        image: bool = True,
        cloud: bool = True,
        semantic: bool = False,
        text: bool = False,
        use_modality_embeddings: bool = False,
        n_heads: int = 4,
        n_layers: int = 4,
        hidden_dim: int = 1024,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.use_modality_embeddings = use_modality_embeddings

        self.modalities = []
        if image:
            self.modalities.append("image")
        if cloud:
            self.modalities.append("cloud")
        if semantic:
            self.modalities.append("semantic")
        if text:
            self.modalities.append("text")

        if self.use_modality_embeddings:
            self.modality_embeddings = nn.ParameterDict(
                {
                    "image": nn.Parameter(torch.randn(desc_dim) * 0.02) if image else None,
                    "cloud": nn.Parameter(torch.randn(desc_dim) * 0.02) if cloud else None,
                    "semantic": nn.Parameter(torch.randn(desc_dim) * 0.02) if semantic else None,
                    "text": nn.Parameter(torch.randn(desc_dim) * 0.02) if text else None,
                }
            )

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=desc_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )

    def forward(self, data: dict[str, Tensor]) -> Tensor:
        descriptors = []

        for key in self.modalities:
            if self.use_modality_embeddings:
                descriptors.append(data[key] + self.modality_embeddings[key])
            else:
                descriptors.append(data[key])

        descriptors = torch.stack(descriptors, dim=1)
        # desc_lens = [d.shape[1] for d in descriptors]
        # descriptors = torch.cat(descriptors, dim=1)
        descriptors = torch.unbind(self.transformer_encoder(descriptors), dim=1)
        # descriptors = torch.split(self.transformer_encoder(descriptors), desc_lens, dim=1)
        out_dict = {}
        for i, key in enumerate(self.modalities):
            out_dict[key] = descriptors[i]
        out_dict["final_descriptor"] = torch.cat(descriptors, dim=-1)
        return out_dict


class SelfAttentionModalityInteraction(nn.Module):
    def __init__(
        self,
        desc_dim: int = 256,
        image: bool = True,
        cloud: bool = True,
        semantic: bool = False,
        text: bool = False,
        use_modality_embeddings: bool = False,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.use_modality_embeddings = use_modality_embeddings

        self.modalities = []
        if image:
            self.modalities.append("image")
        if cloud:
            self.modalities.append("cloud")
        if semantic:
            self.modalities.append("semantic")
        if text:
            self.modalities.append("text")

        if self.use_modality_embeddings:
            self.modality_embeddings = nn.ParameterDict(
                {
                    "image": nn.Parameter(torch.randn(desc_dim) * 0.02) if image else None,
                    "cloud": nn.Parameter(torch.randn(desc_dim) * 0.02) if cloud else None,
                    "semantic": nn.Parameter(torch.randn(desc_dim) * 0.02) if semantic else None,
                    "text": nn.Parameter(torch.randn(desc_dim) * 0.02) if text else None,
                }
            )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=desc_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, data: dict[str, Tensor]) -> Tensor:
        descriptors = []

        for key in self.modalities:
            if self.use_modality_embeddings:
                descriptors.append(data[key] + self.modality_embeddings[key])
            else:
                descriptors.append(data[key])

        # descriptors = torch.stack(descriptors, dim=1)
        # descriptors = torch.unbind(self.self_attention(descriptors, descriptors, descriptors)[0], dim=1)
        desc_lens = [d.shape[1] for d in descriptors]
        descriptors = torch.cat(descriptors, dim=1)
        descriptors = torch.split(
            self.self_attention(descriptors, descriptors, descriptors, need_weights=False)[0],
            desc_lens,
            dim=1,
        )
        out_dict = {}
        for i, key in enumerate(self.modalities):
            out_dict[key] = descriptors[i]
        # out_dict["final_descriptor"] = torch.cat(descriptors, dim=-1)
        return out_dict


class CrossAttentionModalityInteraction(nn.Module):
    def __init__(
        self,
        desc_dim: int = 256,
        image: bool = True,
        cloud: bool = True,
        semantic: bool = False,
        text: bool = False,
        use_modality_embeddings: bool = False,
        n_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.use_modality_embeddings = use_modality_embeddings

        self.modalities = []
        if image:
            self.modalities.append("image")
        if cloud:
            self.modalities.append("cloud")
        if semantic:
            self.modalities.append("semantic")
        if text:
            self.modalities.append("text")

        if self.use_modality_embeddings:
            self.modality_embeddings = nn.ParameterDict(
                {
                    "image": nn.Parameter(torch.randn(desc_dim) * 0.02) if image else None,
                    "cloud": nn.Parameter(torch.randn(desc_dim) * 0.02) if cloud else None,
                    "semantic": nn.Parameter(torch.randn(desc_dim) * 0.02) if semantic else None,
                    "text": nn.Parameter(torch.randn(desc_dim) * 0.02) if text else None,
                }
            )

        self.cross_attn_dict = nn.ModuleDict({})
        for key in self.modalities:
            self.cross_attn_dict[key] = nn.MultiheadAttention(
                embed_dim=desc_dim, num_heads=n_heads, dropout=dropout, batch_first=True
            )

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        out_dict = {}

        for query_modality in self.modalities:
            query = data[query_modality].unsqueeze(1)
            if self.use_modality_embeddings:
                query += self.modality_embeddings[query_modality]

            # Prepare keys and values from other modalities
            keys = []
            for key_modality in self.modalities:
                if key_modality != query_modality:
                    key_value = data[key_modality]
                    if self.use_modality_embeddings:
                        key_value += self.modality_embeddings[key_modality]
                    keys.append(key_value)
            # Stack keys and values from all other modalities
            keys = values = torch.stack(keys, dim=1)

            # Apply cross-attention
            attn_output, _ = self.cross_attn_dict[query_modality](query=query, key=keys, value=values)
            out_dict[query_modality] = attn_output

        out_dict["final_descriptor"] = torch.cat(out_dict.values(), dim=-1)

        return out_dict
