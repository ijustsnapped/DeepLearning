# your_project_name/models/meta_models.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MetadataMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 256, dropout_p: float = 0.4):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)
        
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_p)
        logger.info(f"MetadataMLP initialized: In={input_dim}, Hidden={hidden_dim}, Out={output_dim}, Dropout={dropout_p}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        return x

class CNNWithMetadata(nn.Module):
    def __init__(self,
                 base_cnn_model: nn.Module,
                 num_classes: int,
                 metadata_input_dim: int,
                 meta_mlp_hidden_dim: int = 256,
                 meta_mlp_output_dim: int = 256,
                 meta_dropout_p: float = 0.4,
                 post_concat_dim: int = 1024,
                 post_concat_dropout_p: float = 0.4):
        super().__init__()
        self.base_cnn_model = base_cnn_model
        
        # Get feature dimension from base_cnn_model (assuming timm model)
        # Method 1: If base_cnn_model has a get_classifier method
        if hasattr(self.base_cnn_model, 'get_classifier'):
            self.cnn_feature_dim = self.base_cnn_model.get_classifier().in_features
            # Remove original classifier, features will be output of forward_features or main forward
            self.base_cnn_model.reset_classifier(0, '') # Global pool set to '' if not needed, or specify
        # Method 2: More general, assumes final layer is .fc or .classifier
        elif hasattr(self.base_cnn_model, 'fc'):
            self.cnn_feature_dim = self.base_cnn_model.fc.in_features
            self.base_cnn_model.fc = nn.Identity()
        elif hasattr(self.base_cnn_model, 'classifier'): # e.g. some torchvision models
            # This is trickier as 'classifier' can be a Sequential block
            # For simplicity, let's assume it's a single Linear layer for this example
            if isinstance(self.base_cnn_model.classifier, nn.Linear):
                self.cnn_feature_dim = self.base_cnn_model.classifier.in_features
                self.base_cnn_model.classifier = nn.Identity()
            elif isinstance(self.base_cnn_model.classifier, nn.Sequential):
                 # Try to get in_features from the first linear layer in the Sequential classifier
                for layer in self.base_cnn_model.classifier:
                    if isinstance(layer, nn.Linear):
                        self.cnn_feature_dim = layer.in_features
                        break
                else:
                    raise ValueError("Could not determine CNN feature dimension from Sequential classifier.")
                self.base_cnn_model.classifier = nn.Identity() # Replace the whole block
            else:
                raise ValueError("Unsupported base_cnn_model classifier structure to get feature_dim.")
        else:
            raise ValueError("Cannot determine feature dimension or reset classifier for base_cnn_model.")

        logger.info(f"Base CNN feature dimension: {self.cnn_feature_dim}")

        self.metadata_mlp = MetadataMLP(
            input_dim=metadata_input_dim,
            hidden_dim=meta_mlp_hidden_dim,
            output_dim=meta_mlp_output_dim,
            dropout_p=meta_dropout_p
        )
        
        combined_feature_dim = self.cnn_feature_dim + meta_mlp_output_dim
        
        self.post_concat_fc = nn.Linear(combined_feature_dim, post_concat_dim)
        self.post_concat_bn = nn.BatchNorm1d(post_concat_dim)
        self.post_concat_relu = nn.ReLU()
        self.post_concat_dropout = nn.Dropout(p=post_concat_dropout_p)
        
        self.final_classifier = nn.Linear(post_concat_dim, num_classes)
        
        logger.info(f"CNNWithMetadata initialized: CNN feats={self.cnn_feature_dim}, Meta feats={meta_mlp_output_dim}, "
                    f"Combined={combined_feature_dim}, PostConcat={post_concat_dim}, NumClasses={num_classes}")

    def forward(self, image_input: torch.Tensor, metadata_input: torch.Tensor) -> torch.Tensor:
        # If base_cnn_model.forward_features exists (common in timm)
        if hasattr(self.base_cnn_model, 'forward_features'):
            cnn_features = self.base_cnn_model.forward_features(image_input)
            # Timm models with reset_classifier(0,'') might do GAP in forward_features or need it after
            # For many timm models, after forward_features, if global_pool='', features are [B, C, H, W]
            # If global_pool was set e.g. to 'avg', it's [B,C].
            # Let's ensure it's [B,C] via adaptive avg pooling if needed.
            if cnn_features.ndim == 4: # B, C, H, W
                cnn_features = nn.functional.adaptive_avg_pool2d(cnn_features, (1, 1)).squeeze(-1).squeeze(-1)
        else: # Otherwise, assume self.base_cnn_model(image_input) gives features due to Identity classifier
            cnn_features = self.base_cnn_model(image_input)
            # Output might already be pooled if base_cnn has GAP before Identity, or might need pooling.
            if cnn_features.ndim == 4: # B, C, H, W
                 cnn_features = nn.functional.adaptive_avg_pool2d(cnn_features, (1, 1)).squeeze(-1).squeeze(-1)


        metadata_features = self.metadata_mlp(metadata_input)
        
        combined_features = torch.cat((cnn_features, metadata_features), dim=1)
        
        x = self.post_concat_fc(combined_features)
        x = self.post_concat_bn(x)
        x = self.post_concat_relu(x)
        x = self.post_concat_dropout(x)
        
        output_logits = self.final_classifier(x)
        return output_logits

    def set_base_cnn_trainable(self, trainable: bool):
        for param in self.base_cnn_model.parameters():
            param.requires_grad = trainable
        status = "trainable" if trainable else "frozen"
        logger.info(f"Base CNN model parameters set to {status}.")