#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.modules.crossnet import LowRankCrossNet
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mlp import MLP
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
import numpy as np
import time

def choose(n: int, k: int) -> int:
    """
    Simple implementation of math.comb for Python 3.7 compatibility.
    """
    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


class SparseArch(nn.Module):
    """
    Processes the sparse features of DLRM. Does embedding lookups for all EmbeddingBag
    and embedding features of each collection.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): represents a collection of
            pooled embeddings.

    Example::

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=3, num_embeddings=10, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2", embedding_dim=4, num_embeddings=10, feature_names=["f2"]
        )

        embedding_bag_collection = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        sparse_arch = SparseArch(embedding_bag_collection)

        #     0       1        2  <-- batch
        # 0   [0,1] None    [2]
        # 1   [3]    [4]    [5,6,7]
        # ^
        # feature
        features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f2"],
           values=torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]),
           offsets=torch.tensor([0, 2, 2, 3, 4, 5, 8]),
        )

        sparse_embeddings = sparse_arch(features)
    """

    def __init__(self, embedding_bag_collection: EmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_bag_collection: EmbeddingBagCollection = embedding_bag_collection
        assert (
            self.embedding_bag_collection.embedding_bag_configs
        ), "Embedding bag collection cannot be empty!"
        self.D: int = self.embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs()
            for name in conf.feature_names
        ]

        self.F: int = len(self._sparse_feature_names)

    def forward(
        self,
        features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """

        sparse_features: KeyedTensor = self.embedding_bag_collection(features)

        sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
        sparse_values: List[torch.Tensor] = []
        for name in self.sparse_feature_names:
            sparse_values.append(sparse[name])

        return torch.cat(sparse_values, dim=1).reshape(-1, self.F, self.D)

    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names
    
from torchrec.sparse.tensor_dict import maybe_td_to_kjt
from torchrec.modules.embedding_modules import process_pooled_embeddings, reorder_inverse_indices
import torch.nn.functional as F
torch.fx.wrap('process_pooled_embeddings')
torch.fx.wrap('reorder_inverse_indices')
class __SparseArch(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        use_lora: bool = False,
    ) -> None:
        super().__init__()
        self.embedding_bag_collection = embedding_bag_collection
        assert (
            self.embedding_bag_collection.embedding_bag_configs
        ), "Embedding bag collection cannot be empty!"
        self.D: int = self.embedding_bag_collection.embedding_bag_configs()[0].embedding_dim
        self.configs = embedding_bag_collection.embedding_bag_configs()
        self._sparse_feature_names: List[str] = [
            name
            for conf in embedding_bag_collection.embedding_bag_configs()
            for name in conf.feature_names
        ]
        self.F = len(self._sparse_feature_names)
        self._is_weighted = False
        
        self.use_lora = use_lora
        self.lora_rank = 4
        if self.use_lora:
            assert self.lora_rank > 0, "lora_rank must be positive when use_lora is True"
        
        # Mapping from feature name to table name
        self.feature_to_table = {}
        for config in self.embedding_bag_collection.embedding_bag_configs():
            for feature_name in config.feature_names:
                self.feature_to_table[feature_name] = config.name
        self._feature_names = self.embedding_bag_collection._feature_names
        
        # Initialize LoRA parameters if needed
        if self.use_lora:
            self.lora_As = nn.ModuleDict()
            self.lora_Bs = nn.ModuleDict()
            for config in self.configs:
                table_name = config.name
                original_embedding_bag = self.embedding_bag_collection.embedding_bags[table_name]
                num_embeddings = config.num_embeddings
                embedding_dim = config.embedding_dim
                
                # LoRA A matrix (EmbeddingBag)
                self.lora_As[table_name] = nn.EmbeddingBag(
                    num_embeddings=num_embeddings,
                    embedding_dim=self.lora_rank,
                    mode=original_embedding_bag.mode,
                    include_last_offset=original_embedding_bag.include_last_offset
                )
                nn.init.zeros_(self.lora_As[table_name].weight)
                
                # LoRA B matrix (EmbeddingBag)
                self.lora_Bs[table_name] = nn.EmbeddingBag(
                    num_embeddings=self.lora_rank,  # 目标维度
                    embedding_dim=embedding_dim,  # 输入维度
                    mode='sum',  # 或者根据需求选择 'mean' 或 'max'
                    include_last_offset=False  # 根据需求设置
                )
                nn.init.normal_(self.lora_Bs[table_name].weight, mean=0.0, std=0.01)
            
            # Freeze original embedding bags
            for param in self.embedding_bag_collection.parameters():
                param.requires_grad = False

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        """
        Args:
            features (KeyedJaggedTensor): an input tensor of sparse features.

        Returns:
            torch.Tensor: tensor of shape B X F X D.
        """

        # Get the sparse features from the embedding bag collection
        sparse_features: KeyedTensor = self.embedding_bag_collection(features)
        sparse: Dict[str, torch.Tensor] = sparse_features.to_dict()
        sparse_values: List[torch.Tensor] = []

        for name in self._sparse_feature_names:
            feature_tensor = sparse[name]

            if self.use_lora:
                # Get the table name associated with the feature
                table_name = self.feature_to_table[name]
                
                # Apply LoRA A transformation using F.embedding_bag
                flat_feature_names: List[str] = []
                features = maybe_td_to_kjt(features, None)
                for names in self._feature_names:
                    flat_feature_names.extend(names)
                inverse_indices = reorder_inverse_indices(
                    inverse_indices=features.inverse_indices_or_none(),
                    feature_names=flat_feature_names,
                )
                feature_dict = features.to_dict()
                print(f"feature_dict:{feature_dict}")
                print(f"name:{name}")
                f = feature_dict[name]
                lora_A_output = self.lora_As[table_name](
                    input=f.values(),
                    offsets=f.offsets(),
                    per_sample_weights=(
                        f.weights().to(self.lora_As[table_name].weight.dtype)
                        if self._is_weighted
                        else None
                    ),
                ).float()
                lora_A_output=process_pooled_embeddings(
                    pooled_embeddings=lora_A_output,
                    inverse_indices=inverse_indices,
                ),
                    
                # Apply LoRA B transformation
                lora_B_output = self.lora_Bs[table_name]
                feature_tensor += (lora_A_output @ lora_B_output.weight)

            sparse_values.append(feature_tensor)

        # Concatenate and reshape the resulting tensor
        return torch.cat(sparse_values, dim=1).reshape(-1, self.F, self.D)
    
    @property
    def sparse_feature_names(self) -> List[str]:
        return self._sparse_feature_names

class DenseArch(nn.Module):
    """
    Processes the dense features of DLRM model.

    Args:
        in_features (int): dimensionality of the dense input features.
        layer_sizes (List[int]): list of layer sizes.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        dense_arch = DenseArch(10, layer_sizes=[15, D])
        dense_embedded = dense_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.model: nn.Module = MLP(
            in_features, layer_sizes, bias=True, activation="relu", device=device
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor): an input tensor of dense features.

        Returns:
            torch.Tensor: an output tensor of size B X D.
        """
        return self.model(features)


class InteractionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the pairwise dot product of each sparse feature pair,
    the dot product of each sparse features with the output of the dense layer,
    and the dense layer itself (all concatenated).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        inter_arch = InteractionArch(num_sparse_features=len(keys))

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F + F choose 2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: int) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.register_buffer(
            "triu_indices",
            torch.triu_indices(self.F + 1, self.F + 1, offset=1),
            persistent=False,
        )

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (D + F + F choose 2).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )

        # dense/sparse + sparse/sparse interaction
        # size B X (F + F choose 2)
        interactions = torch.bmm(
            combined_values, torch.transpose(combined_values, 1, 2)
        )
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return torch.cat((dense_features, interactions_flat), dim=1)


class InteractionDCNArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Returns the output of a Deep Cross Net v2
    https://arxiv.org/pdf/2008.13535.pdf with a low rank approximation for the
    weight matrix. The input and output sizes are the same for this
    interaction layer (F*D + D).

    .. note::
        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.


    Args:
        num_sparse_features (int): F.

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        DCN = LowRankCrossNet(
            in_features = F*D+D,
            dcn_num_layers = 2,
            dnc_low_rank_dim = 4,
        )
        inter_arch = InteractionDCNArch(
            num_sparse_features=len(keys),
            crossnet=DCN,
        )

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (F*D + D)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(self, num_sparse_features: int, crossnet: nn.Module) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.crossnet = crossnet

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (F*D + D).
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )

        # size B X (F*D + D)
        return self.crossnet(combined_values.reshape([B, -1]))


class InteractionProjectionArch(nn.Module):
    """
    Processes the output of both `SparseArch` (sparse_features) and `DenseArch`
    (dense_features). Return Y*Z and the dense layer itself (all concatenated)
    where Y is the output of interaction branch 1 and Z is the output of interaction
    branch 2. Y and Z are of size Bx(F1xD) and Bx(DxF2) respectively for some F1 and F2.

    .. note::

        The dimensionality of the `dense_features` (D) is expected to match the
        dimensionality of the `sparse_features` so that the dot products between them
        can be computed.
        The output dimension of the 2 interaction branches should be a multiple
        of D.


    Args:
        num_sparse_features (int): F.
        interaction_branch1 (nn.Module): MLP module for the first branch of
            interaction layer
        interaction_branch2 (nn.Module): MLP module for the second branch of
            interaction layer

    Example::

        D = 3
        B = 10
        keys = ["f1", "f2"]
        F = len(keys)
        # Assume last layer of
        I1 = DenseArch(
            in_features= 3 * D + D,
            layer_sizes=[4*D, 4*D], # F1 = 4
            device=dense_device,
        )
        I2 = DenseArch(
            in_features= 3 * D + D,
            layer_sizes=[4*D, 4*D], # F2 = 4
            device=dense_device,
        )
        inter_arch = InteractionProjectionArch(
                        num_sparse_features=len(keys),
                        interaction_branch1 = I1,
                        interaction_branch2 = I2,
                    )

        dense_features = torch.rand((B, D))
        sparse_features = torch.rand((B, F, D))

        #  B X (D + F1 * F2)
        concat_dense = inter_arch(dense_features, sparse_features)
    """

    def __init__(
        self,
        num_sparse_features: int,
        interaction_branch1: nn.Module,
        interaction_branch2: nn.Module,
    ) -> None:
        super().__init__()
        self.F: int = num_sparse_features
        self.interaction_branch1 = interaction_branch1
        self.interaction_branch2 = interaction_branch2

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X F X D.

        Returns:
            torch.Tensor: an output tensor of size B X (D + F1 * F2)) where
            F1*D and F2*D are the output dimensions of the 2 interaction MLPs.
        """
        if self.F <= 0:
            return dense_features
        (B, D) = dense_features.shape

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )

        interaction_branch1_out = self.interaction_branch1(
            torch.reshape(combined_values, (B, -1))
        )

        interaction_branch2_out = self.interaction_branch2(
            torch.reshape(combined_values, (B, -1))
        )

        interactions = torch.bmm(
            interaction_branch1_out.reshape([B, -1, D]),
            interaction_branch2_out.reshape([B, D, -1]),
        )
        interactions_flat = torch.reshape(interactions, (B, -1))

        return torch.cat((dense_features, interactions_flat), dim=1)


class OverArch(nn.Module):
    """
    Final Arch of DLRM - simple MLP over OverArch.

    Args:
        in_features (int): size of the input.
        layer_sizes (List[int]): sizes of the layers of the `OverArch`.
        device (Optional[torch.device]): default compute device.

    Example::

        B = 20
        D = 3
        over_arch = OverArch(10, [5, 1])
        logits = over_arch(torch.rand((B, 10)))
    """

    def __init__(
        self,
        in_features: int,
        layer_sizes: List[int],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        if len(layer_sizes) <= 1:
            raise ValueError("OverArch must have multiple layers.")
        self.model: nn.Module = nn.Sequential(
            MLP(
                in_features,
                layer_sizes[:-1],
                bias=True,
                activation="relu",
                device=device,
            ),
            nn.Linear(layer_sizes[-2], layer_sizes[-1], bias=True, device=device),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (torch.Tensor):

        Returns:
            torch.Tensor: size B X layer_sizes[-1]
        """
        return self.model(features)

import math
class DLRM(nn.Module):
    """
    Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        dense_device (Optional[torch.device]): default compute device.

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
            name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1"]
        )
        eb2_config = EmbeddingBagConfig(
            name="t2",
            embedding_dim=D,
            num_embeddings=100,
            feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM(
            embedding_bag_collection=ebc,
            dense_in_features=100,
            dense_arch_layer_sizes=[20, D],
            over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["f1", "f2"],
            values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
            offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
            dense_features=features,
            sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
        use_lora: bool = False
    ) -> None:
        super().__init__()
        assert (
            len(embedding_bag_collection.embedding_bag_configs()) > 0
        ), "At least one embedding bag is required"
        for i in range(1, len(embedding_bag_collection.embedding_bag_configs())):
            conf_prev = embedding_bag_collection.embedding_bag_configs()[i - 1]
            conf = embedding_bag_collection.embedding_bag_configs()[i]
            assert (
                conf_prev.embedding_dim == conf.embedding_dim
            ), "All EmbeddingBagConfigs must have the same dimension"
        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError(
                f"embedding_bag_collection dimension ({embedding_dim}) and final dense "
                "arch layer size ({dense_arch_layer_sizes[-1]}) must match."
            )

        self.sparse_arch: SparseArch = SparseArch(embedding_bag_collection)
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        self.dense_arch = DenseArch(
            in_features=dense_in_features,
            layer_sizes=dense_arch_layer_sizes,
            device=dense_device,
        )

        self.inter_arch = InteractionArch(
            num_sparse_features=num_sparse_features,
        )

        over_in_features: int = (
            embedding_dim + choose(num_sparse_features, 2) + num_sparse_features
        )

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )
        if use_lora:
            first_embedding = next(iter(self.sparse_arch.embedding_bag_collection.embedding_bags.values()))
            scale_grad_A = math.sqrt(first_embedding._r / embedding_dim)
            for config in embedding_bag_collection.embedding_bag_configs():
                    for feature_name in config.feature_names:
                        self.sparse_arch.embedding_bag_collection.embedding_bags[f"t_{feature_name}"]._A.register_hook(lambda grad: grad * scale_grad_A)
            for module in [self.dense_arch, self.inter_arch, self.over_arch]:
                    for param in module.parameters():
                        param.requires_grad = False

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
            dense_features (torch.Tensor): the dense features.
            sparse_features (KeyedJaggedTensor): the sparse features.

        Returns:
            torch.Tensor: logits.
        """
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)

        concatenated_dense = self.inter_arch(
            dense_features=embedded_dense, sparse_features=embedded_sparse
        )
        logits = self.over_arch(concatenated_dense)
        return logits

from collections import defaultdict
from torch.fx.proxy import Proxy
class __DLRM(nn.Module):
    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
        use_lora: bool = False,
        lora_rank: int = 8,
    ) -> None:
        super().__init__()
        # 原始参数校验保持不变
        assert len(embedding_bag_collection.embedding_bag_configs()) > 0, "需要至少一个EmbeddingBag"
        embedding_dim = embedding_bag_collection.embedding_bag_configs()[0].embedding_dim
        for config in embedding_bag_collection.embedding_bag_configs()[1:]:
            assert config.embedding_dim == embedding_dim, "所有EmbeddingBag维度需一致"
        if dense_arch_layer_sizes[-1] != embedding_dim:
            raise ValueError("Dense输出维度需与Embedding维度一致")

        # 初始化模块
        self.sparse_arch = SparseArch(embedding_bag_collection)
        self.dense_arch = DenseArch(dense_in_features, dense_arch_layer_sizes, dense_device)
        self.inter_arch = InteractionArch(len(self.sparse_arch.sparse_feature_names))
        over_in_features = embedding_dim + (len(self.sparse_arch.sparse_feature_names) * (len(self.sparse_arch.sparse_feature_names) + 1)) // 2
        self.over_arch = OverArch(over_in_features, over_arch_layer_sizes, dense_device)
        
        # LoRA相关初始化
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_rank = lora_rank
            self.feature_to_table = {}
            self.table_indices = defaultdict(set)
            self.lora_As = nn.ModuleDict()
            self.lora_Bs = nn.ModuleDict()
            
            # 构建特征到表的映射
            for config in embedding_bag_collection.embedding_bag_configs():
                for feature_name in config.feature_names:
                    self.feature_to_table[feature_name] = config.name
                # 初始化LoRA参数
                table_name = config.name
                original_embedding_bag = embedding_bag_collection.embedding_bags[table_name]
                # LoRA A使用与原EmbeddingBag相同的配置
                self.lora_As[table_name] = nn.EmbeddingBag(
                    num_embeddings=config.num_embeddings,
                    embedding_dim=lora_rank,
                    mode=original_embedding_bag.mode,
                    include_last_offset=original_embedding_bag.include_last_offset
                )
                # nn.init.normal_(self.lora_As[table_name].weight, mean=0.0, std=0.01)
                nn.init.zeros_(self.lora_As[table_name].weight)
                # LoRA B使用线性层
                self.lora_Bs[table_name] = nn.Linear(embedding_dim, lora_rank, bias=False)
                nn.init.normal_(self.lora_Bs[table_name].weight, mean=0.0, std=0.01)
            
            # 冻结原始参数
            for module in [self.sparse_arch, self.dense_arch, self.inter_arch, self.over_arch]:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(
        self,
        dense_features: torch.Tensor,
        sparse_features: KeyedJaggedTensor,
    ) -> torch.Tensor:
        embedded_dense = self.dense_arch(dense_features)
        embedded_sparse = self.sparse_arch(sparse_features)  # [B, F, D]
        
        if self.use_lora:
            batch_size, num_features, emb_dim = embedded_sparse.shape
            lora_adjustments = []
            
            for feature_name in self.sparse_arch.sparse_feature_names:
                table_name = self.feature_to_table[feature_name]
                feature_data = sparse_features[feature_name]
                indices = feature_data.values()
                offsets = feature_data.offsets()

                # 训练时记录索引
                if self.training and not isinstance(indices, Proxy):
                    self.table_indices[table_name].update(indices.cpu().numpy().tolist())
                
                if self.training:
                    # 通过LoRA A获取低秩表示
                    lora_A_output = self.lora_As[table_name](
                        input=indices,
                        offsets=offsets,
                        per_sample_weights=feature_data.weights_or_none()
                    )  # 形状: [B, lora_rank]
                else:
                    if not isinstance(indices, Proxy):
                        # print(f"indices shape:{indices.shape}")

                        # 获取已记录索引
                        recorded = self.table_indices.get(table_name, set())
                        # print(f"recorded.shape:{len(recorded)}")  # set 没有 shape，改为 len()

                        if not recorded:
                            lora_adjustments.append(torch.zeros(batch_size, 1, emb_dim, device=embedded_sparse.device))
                            continue

                        # 生成索引掩码（True 表示需要保留，False 表示要置零）
                        mask = torch.isin(indices, torch.tensor(list(recorded), device=indices.device))

                        # 计算 LoRA A 的输出
                        lora_A_output = self.lora_As[table_name](
                            indices,  # 直接使用原始 indices
                            offsets,
                            feature_data.weights_or_none() if feature_data.weights_or_none() is not None else None
                        )

                        # 直接用 mask 过滤输出
                        lora_A_output *= mask.view(-1, 1)  # 适配 [2048, 4]

                # 通过LoRA B投影回原始空间
                lora_B_output = lora_A_output @ self.lora_Bs[table_name].weight  # 形状: [B, D]
                lora_B_output = torch.zeros_like(lora_B_output)

                # 将调整量添加到列表
                lora_adjustments.append(lora_B_output.unsqueeze(1))  # [B, 1, D]
            # 合并所有调整量
            lora_adjustment = torch.cat(lora_adjustments, dim=1)  # [B, F, D]
            
            # 应用调整
            embedded_sparse += lora_adjustment
        
        # 后续处理
        concatenated = self.inter_arch(embedded_dense, embedded_sparse)
        return self.over_arch(concatenated)

class DLRM_Projection(DLRM):
    """
    Recsys model modified from the original model from "Deep Learning Recommendation
    Model for Personalization and Recommendation Systems"
    (https://arxiv.org/abs/1906.00091). Similar to DLRM module but has
    additional MLPs in the interaction layer (along 2 branches).

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        interaction_branch1_layer_sizes (List[int]): the layer sizes for first branch of
            interaction layer. The output dimension must be a multiple of D.
        interaction_branch2_layer_sizes (List[int]):the layer sizes for second branch of
            interaction layer. The output dimension must be a multiple of D.
        dense_device (Optional[torch.device]): default compute device.

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM_Projection(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20, D],
           interaction_branch1_layer_sizes=[3*D+D, 4*D],
           interaction_branch2_layer_sizes=[3*D+D, 4*D],
           over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        interaction_branch1_layer_sizes: List[int],
        interaction_branch2_layer_sizes: List[int],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        # initialize DLRM
        # sparse arch and dense arch are initialized via DLRM
        super().__init__(
            embedding_bag_collection,
            dense_in_features,
            dense_arch_layer_sizes,
            over_arch_layer_sizes,
            dense_device,
        )

        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        # Fix interaction and over arch for DLRM_Projeciton
        if interaction_branch1_layer_sizes[-1] % embedding_dim != 0:
            raise ValueError(
                "Final interaction branch1 layer size "
                "({}) is not a multiple of embedding size ({})".format(
                    interaction_branch1_layer_sizes[-1], embedding_dim
                )
            )
        projected_dim_1: int = interaction_branch1_layer_sizes[-1] // embedding_dim
        interaction_branch1 = DenseArch(
            in_features=num_sparse_features * embedding_dim
            + dense_arch_layer_sizes[-1],
            layer_sizes=interaction_branch1_layer_sizes,
            device=dense_device,
        )

        if interaction_branch2_layer_sizes[-1] % embedding_dim != 0:
            raise ValueError(
                "Final interaction branch2 layer size "
                "({}) is not a multiple of embedding size ({})".format(
                    interaction_branch2_layer_sizes[-1], embedding_dim
                )
            )
        projected_dim_2: int = interaction_branch2_layer_sizes[-1] // embedding_dim
        interaction_branch2 = DenseArch(
            in_features=num_sparse_features * embedding_dim
            + dense_arch_layer_sizes[-1],
            layer_sizes=interaction_branch2_layer_sizes,
            device=dense_device,
        )

        self.inter_arch = InteractionProjectionArch(
            num_sparse_features=num_sparse_features,
            interaction_branch1=interaction_branch1,
            interaction_branch2=interaction_branch2,
        )

        over_in_features: int = embedding_dim + projected_dim_1 * projected_dim_2

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )


class DLRM_DCN(DLRM):
    """
    Recsys model with DCN modified from the original model from "Deep Learning Recommendation
    Model for Personalization and Recommendation Systems"
    (https://arxiv.org/abs/1906.00091). Similar to DLRM module but has
    DeepCrossNet https://arxiv.org/pdf/2008.13535.pdf as the interaction layer.

    The module assumes all sparse features have the same embedding dimension
    (i.e. each EmbeddingBagConfig uses the same embedding_dim).

    The following notation is used throughout the documentation for the models:

    * F: number of sparse features
    * D: embedding_dimension of sparse features
    * B: batch size
    * num_features: number of dense features

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define `SparseArch`.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the `DenseArch`.
        over_arch_layer_sizes (List[int]): the layer sizes for the `OverArch`.
            The output dimension of the `InteractionArch` should not be manually
            specified here.
        dcn_num_layers (int): the number of DCN layers in the interaction.
        dcn_low_rank_dim (int): the dimensionality of low rank approximation
            used in the dcn layers.
        dense_device (Optional[torch.device]): default compute device.

    Example::

        B = 2
        D = 8

        eb1_config = EmbeddingBagConfig(
           name="t1", embedding_dim=D, num_embeddings=100, feature_names=["f1", "f3"]
        )
        eb2_config = EmbeddingBagConfig(
           name="t2",
           embedding_dim=D,
           num_embeddings=100,
           feature_names=["f2"],
        )

        ebc = EmbeddingBagCollection(tables=[eb1_config, eb2_config])
        model = DLRM_DCN(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20, D],
           dcn_num_layers=2,
           dcn_low_rank_dim=8,
           over_arch_layer_sizes=[5, 1],
        )

        features = torch.rand((B, 100))

        #     0       1
        # 0   [1,2] [4,5]
        # 1   [4,3] [2,9]
        # ^
        # feature
        sparse_features = KeyedJaggedTensor.from_offsets_sync(
           keys=["f1", "f3"],
           values=torch.tensor([1, 2, 4, 5, 4, 3, 2, 9]),
           offsets=torch.tensor([0, 2, 4, 6, 8]),
        )

        logits = model(
           dense_features=features,
           sparse_features=sparse_features,
        )
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        dcn_num_layers: int,
        dcn_low_rank_dim: int,
        dense_device: Optional[torch.device] = None,
    ) -> None:
        # initialize DLRM
        # sparse arch and dense arch are initialized via DLRM
        super().__init__(
            embedding_bag_collection,
            dense_in_features,
            dense_arch_layer_sizes,
            over_arch_layer_sizes,
            dense_device,
        )

        embedding_dim: int = embedding_bag_collection.embedding_bag_configs()[
            0
        ].embedding_dim
        num_sparse_features: int = len(self.sparse_arch.sparse_feature_names)

        # Fix interaction and over arch for DLRM_DCN

        crossnet = LowRankCrossNet(
            in_features=(num_sparse_features + 1) * embedding_dim,
            num_layers=dcn_num_layers,
            low_rank=dcn_low_rank_dim,
        )

        self.inter_arch = InteractionDCNArch(
            num_sparse_features=num_sparse_features,
            crossnet=crossnet,
        )

        over_in_features: int = (num_sparse_features + 1) * embedding_dim

        self.over_arch = OverArch(
            in_features=over_in_features,
            layer_sizes=over_arch_layer_sizes,
            device=dense_device,
        )


class DLRMTrain(nn.Module):
    """
    nn.Module to wrap DLRM model to use with train_pipeline.

    DLRM Recsys model from "Deep Learning Recommendation Model for Personalization and
    Recommendation Systems" (https://arxiv.org/abs/1906.00091). Processes sparse
    features by learning pooled embeddings for each feature. Learns the relationship
    between dense features and sparse features by projecting dense features into the
    same embedding space. Also, learns the pairwise relationships between sparse
    features.

    The module assumes all sparse features have the same embedding dimension
    (i.e, each EmbeddingBagConfig uses the same embedding_dim)

    Args:
        dlrm_module: DLRM module (DLRM or DLRM_Projection or DLRM_DCN) to be used in
        training

    Example::

        ebc = EmbeddingBagCollection(config=ebc_config)
        dlrm_module = DLRM(
           embedding_bag_collection=ebc,
           dense_in_features=100,
           dense_arch_layer_sizes=[20],
           over_arch_layer_sizes=[5, 1],
        )
        dlrm_model = DLRMTrain(dlrm_module)
    """

    def __init__(
        self,
        dlrm_module: DLRM,
    ) -> None:
        super().__init__()
        self.model = dlrm_module
        self.loss_fn: nn.Module = nn.BCEWithLogitsLoss()

    def forward(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            batch: batch used with criteo and random data from torchrec.datasets
        Returns:
            Tuple[loss, Tuple[loss, logits, labels]]
        """
        logits = self.model(batch.dense_features, batch.sparse_features)
        logits = logits.squeeze(-1)
        loss = self.loss_fn(logits, batch.labels.float())

        return loss, (loss.detach(), logits.detach(), batch.labels.detach())
