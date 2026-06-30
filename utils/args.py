"""args.py

Command-line argument definitions for the training / evaluation pipeline.
"""

import argparse


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prototype-based classification with configurable backbones"
    )
    # General arguments
    parser.add_argument(
        '--output_dir',
        type=str,
        default="../OUTPUTS_OPAL",   
        help='Directory to save the model and other outputs'
    )
    parser.add_argument(
        '--config_dir',
        type=str,
        default=None,
        help='Directory where all the arguments are in a .txt' 
    )
    
    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default='CUB-200-2011',
        help='Dataset to be used for train and test'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/CUB_200_2011/dataset',
        help='Directory containing the dataset. Should contain train and test folders'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=224,
        help="Input images will be resized to --img_size"
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=200,
        help='Number of classes to use in the classification. If 0, it will be inferred from the dataset. Else it will use the first X classes'
    )
    parser.add_argument(
        '--data_aug',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Whether to use data augmentation or not'
    )
    
    # Model arguments
    parser.add_argument(
        '--base_model',
        type=str,
        default='convnext_tiny',
        help='Define feature extractor model to use. If it is not one of the existing models, it must be included in the code.'
    )
    parser.add_argument(
        '--pretrain',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Define whether the feature extractor is initialized with pre-trained weights or not'
    )
    parser.add_argument(
        '--pooling',
        type=str,
        default='max',
        choices=['avg', 'max'],
        help="Pooling to be done for the features maps"
    )
    parser.add_argument(
        '--classifier',
        type=str,
        default='norm_prototypes_dist',
        choices=['linear', 'norm_prototypes_dist'],
        help='Classification head to use'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='euclidean',
        choices=['lp', 'euclidean', 'chebyshev', 'tropical', 'kl'],
        help='Distance metric to be used for the logits')
    parser.add_argument(
        '--p',
        type=int,
        default=2,
        choices=[1, 2, 3],
        help='Distance metric to be used for the logits. 1: Manhattan, 2: Euclidean, 3:...')
    parser.add_argument(
        '--ch_last_conv',
        type=int,
        default=5,
        help='Define how many convolutional filters you want to have in the last layer. If 0, the number of convolutional filters in the last layer is not modified.'
    )
    parser.add_argument(
        '--train',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Define if we are going to train a model or load a trained one'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='NNModel_softmax',
        choices=['NNModel', 'NNModel_softmax'],
        help='Model variant to use: NNModel (no activation) or NNModel_softmax (channel softmax)'
    )
    
    # Model Hyperparameters
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility. Could be some differences between runs due to nondeterminism'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Number of epochs to train'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='images per batch'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate value'
    )
    parser.add_argument(
        '--lr_scheduler',
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help='Define if learning scheduler is going to be used'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='Adam',
        help='Optimizer used for training'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0,
        help='Weight decay used in the optimizer'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers in dataloader'
    )
    parser.add_argument(
        '--pin_memory',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Copy tensors to CUDA memory'
    )
    
    # Results and outputs arguments
    parser.add_argument(
        '--save_model',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Save best model'
    )
    parser.add_argument(
        '--save_metrics',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Save quantitative metrics'
    )
    parser.add_argument(
        '--save_curves',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Save training and validation curves'
    )
    parser.add_argument(
        '--save_tsne',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Save tsne embedding representation'
    )
    parser.add_argument(
        '--save_arch_imgs',
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help='Compute and save archetypal images'
    )
    parser.add_argument(
        '--save_arch_proto',
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help='Compute and save prototypes of archetypal images'
    )
    parser.add_argument(
        '--save_collapse_metrics',
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help='Compute and save Spatial Prototype Collapse (SPC) metrics'
    )
    
    args = parser.parse_args()
    return args