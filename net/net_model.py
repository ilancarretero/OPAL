"""
net_model.py

Neural network model architectures combining a feature extractor backbone,
an optional final convolutional layer, pooling, normalization, and a
classification head. Two variants are provided:

  - NNModel:         No activation before pooling.
  - NNModel_softmax: Channel-wise softmax before pooling.
"""

import torch
import torch.nn as nn
from .baseline import get_feature_extractor, get_final_conv_layer, get_pooling
from .classifiers import get_classifier


# ---------------------------------------------------------------------------
# Base model: no activation before pooling
# ---------------------------------------------------------------------------
class NNModel(nn.Module):
    def __init__(self, args):
        """
        Args:
            args (argparse.Namespace): Arguments containing:
                - base_model (str): Name of the base model, e.g., "convnext_tiny", "resnet50", etc.
                - pretrain (bool): Whether to use pretrained weights.
                - pooling (str): Type of pooling ("avg" or "max").
                - classifier (str): Name of the classifier ("linear", "prototype", etc.)
                - num_classes (int): Number of output classes.
        """
        super(NNModel, self).__init__()
        # Get the feature extractor (backbone)
        self.backbone = get_feature_extractor(args)
        # Get the adaptive pooling layer
        self.pooling = get_pooling(args)
        
        # Infer the embedding dimension (number of channels) using a dummy input.
        dummy_input = torch.randn(1, 3, 224, 224)  # Adjust size if necessary.
        with torch.no_grad():
            features = self.backbone(dummy_input)
        # features.shape is [1, channels, H, W]
        self.embedding_dim = features.shape[1]

        # Get the final convolutional layer
        self.final_conv_layer = get_final_conv_layer(args, in_channels=self.embedding_dim)
        # Get number of final convolutional filters
        if args.ch_last_conv == 0:
            self.final_num_convs = self.embedding_dim
        else:
            self.final_num_convs = args.num_classes * args.ch_last_conv
        # Get the classifier with the inferred embedding_dim
        self.classifier = get_classifier(args, self.final_num_convs)

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        features = self.backbone(x)        # Extract the feature map
        conv_features = self.final_conv_layer(features)
        pooled = self.pooling(conv_features)      # Apply pooling to get [B, channels, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # Flatten to [B, channels]
        # Normalize the embedding to improve stability
        norm = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-8)
        # Calculate logits with the classifier
        logits = self.classifier(norm)
        return logits
    
    def get_embedding_train(self, x):
        features = self.backbone(x)
        conv_features = self.final_conv_layer(features)
        embeddings = self.pooling(conv_features)
        embeddings = embeddings.view(embeddings.size(0), -1)
        embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        return embeddings
    
    # Function to extract embeddings from the backbone (after pooling, before FC)
    def get_embedding(self, x):
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            conv_features = self.final_conv_layer(features)
            embeddings = self.pooling(conv_features)
            embeddings = embeddings.view(embeddings.size(0), -1)
            embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        return embeddings

    def get_last_conv(self, x):
        """Return the spatial feature map [B, C, H, W] after the final conv layer (before pooling)."""
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            conv_features = self.final_conv_layer(features)
        return conv_features


# ---------------------------------------------------------------------------
# Softmax variant: channel-wise softmax before pooling
# ---------------------------------------------------------------------------
class NNModel_softmax(nn.Module):
    def __init__(self, args):
        """
        Args:
            args (argparse.Namespace): Arguments containing:
                - base_model (str): Name of the base model, e.g., "convnext_tiny", "resnet50", etc.
                - pretrain (bool): Whether to use pretrained weights.
                - pooling (str): Type of pooling ("avg" or "max").
                - classifier (str): Name of the classifier ("linear", "prototype", etc.)
                - num_classes (int): Number of output classes.
        """
        super(NNModel_softmax, self).__init__()
        # Get the feature extractor (backbone)
        self.backbone = get_feature_extractor(args)
        # Get the adaptive pooling layer
        self.pooling = get_pooling(args)
        # Add sofmax per channel
        self.softmax = nn.Softmax(dim=1)
        
        # Infer the embedding dimension (number of channels) using a dummy input.
        dummy_input = torch.randn(1, 3, 224, 224)  # Adjust size if necessary.
        with torch.no_grad():
            features = self.backbone(dummy_input)
        # features.shape is [1, channels, H, W]
        self.embedding_dim = features.shape[1]

        # Get the final convolutional layer
        self.final_conv_layer = get_final_conv_layer(args, in_channels=self.embedding_dim)
        # Get number of final convolutional filters
        if args.ch_last_conv == 0:
            self.final_num_convs = self.embedding_dim
        else:
            self.final_num_convs = args.num_classes * args.ch_last_conv
        # Get the classifier with the inferred embedding_dim
        self.classifier = get_classifier(args, self.final_num_convs)

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        features = self.backbone(x)        # Extract the feature map
        conv_features = self.final_conv_layer(features)
        prob_features = self.softmax(conv_features)  # Apply softmax to get probabilities
        pooled = self.pooling(prob_features)      # Apply pooling to get [B, channels, 1, 1]
        pooled = pooled.view(pooled.size(0), -1)  # Flatten to [B, channels]
        # Normalize the embedding to improve stability
        norm = pooled / (pooled.norm(dim=1, keepdim=True) + 1e-8)
        # Calculate logits with the classifier
        logits = self.classifier(norm)
        return logits
    
    def get_embedding_train(self, x):
        features = self.backbone(x)
        conv_features = self.final_conv_layer(features)
        prob_features = self.softmax(conv_features)
        embeddings = self.pooling(prob_features)
        embeddings = embeddings.view(embeddings.size(0), -1)
        embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        return embeddings
    
    # Function to extract embeddings from the backbone (after pooling, before FC)
    def get_embedding(self, x):
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            conv_features = self.final_conv_layer(features)
            prob_features = self.softmax(conv_features)
            embeddings = self.pooling(prob_features)
            embeddings = embeddings.view(embeddings.size(0), -1)
            embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + 1e-8)
        return embeddings
    
    def get_last_conv(self, x):
        """Return the spatial probability map [B, C, H, W] after softmax (before pooling)."""
        self.eval()
        with torch.no_grad():
            features = self.backbone(x)
            conv_features = self.final_conv_layer(features)
            prob_features = self.softmax(conv_features)
        return prob_features