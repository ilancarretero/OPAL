"""classifiers.py

Classification heads: prototype-based and linear classifiers.
"""

import argparse

import torch
import torch.nn as nn


class PrototypeClassifierNormDist(torch.nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int,
                 metric: str = 'euclidean', p: float = 2.0,
                 sw_projections: int = 50):
        """
        A prototype-based classifier with flexible distance/similarity.

        Args:
            num_classes (int):      Number of classes/prototypes.
            embedding_dim (int):    Dimensionality of each embedding.
            metric (str):           One of {
                                     'euclidean','lp','chebyshev',
                                     'tropical','cosine',
                                     'kl','emd','wasserstein','js'
                                   }.
                                     - 'euclidean':   L2 distance
                                     - 'lp':          Lp distance with user p
                                     - 'chebyshev':   Linf distance
                                     - 'tropical':    tropical distance
                                     - 'kl':          KL-divergence on histograms
            p (float):              Exponent for 'lp' metric (ignored otherwise).
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.metric = metric.lower()
        self.p = p

        # Build prototypes: one-hot blocks (normalized to unit norm)
        channels_per = embedding_dim // num_classes
        protos = torch.zeros(num_classes, embedding_dim)
        for c in range(num_classes):
            start = c * channels_per
            end = (c + 1) * channels_per if c < num_classes - 1 else embedding_dim
            protos[c, start:end] = 1.0
        protos = protos / protos.norm(dim=1, keepdim=True)
        self.register_buffer('prototypes', protos)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, embedding_dim).
        For 'kl' should be non-negative.

        Returns:
            logits: (B, num_classes), higher = more likely class.
        """
        if self.metric in ('euclidean', 'lp', 'chebyshev'):
            # Lp / Linf distances
            if self.metric == 'euclidean':
                p_val = 2.0
            elif self.metric == 'chebyshev':
                p_val = float('inf')
            else:
                p_val = self.p
            dists = torch.cdist(embeddings, self.prototypes, p=p_val)
            return -dists

        elif self.metric == 'tropical':
            # Tropical distance: max minus min of coordinate diffs
            diff = embeddings.unsqueeze(1) - self.prototypes.unsqueeze(0)
            mx = diff.max(dim=2).values
            mn = diff.min(dim=2).values
            trop = mx - mn
            return -trop

        # For histogram-based divergences/distances:
        eps = 1e-8
        x_exp = embeddings.unsqueeze(1)  # (B,1,D)
        y_exp = self.prototypes.unsqueeze(0)  # (1,C,D)

        if self.metric == 'kl':
            # Kullback-Leibler divergence
            kl = torch.sum(x_exp * (torch.log(x_exp + eps) - torch.log(y_exp + eps)), dim=2)
            return -kl

        else:
            raise ValueError(f"Unknown metric: {self.metric!r}")


def get_classifier(args: argparse.Namespace, embedding_dim: int) -> torch.nn.Module:
    """
    Instantiate and return a classifier based on the given arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments containing:
            - classifier (str): Name of the classifier to select.
            - num_classes (int): Number of classes.
        embedding_dim (int): Dimension of the embeddings.
        
    Returns:
        nn.Module: An instance of the selected classifier.
        
    Raises:
        ValueError: If an unsupported classifier name is provided.
    """
    # Convert classifier name to lowercase for case-insensitive matching.
    classifier_key = args.classifier.lower()
    
    # If the user selects 'linear', return a linear layer from embedding_dim to num_classes.
    if classifier_key == "linear":
        return torch.nn.Linear(embedding_dim, args.num_classes)
    
    if classifier_key == "norm_prototypes_dist":
        return PrototypeClassifierNormDist(args.num_classes, embedding_dim,
                                           metric=args.metric, p=args.p)

    raise ValueError(
        f"Unsupported classifier: {args.classifier!r}. "
        f"Choose from: 'linear', 'norm_prototypes_dist'."
    )