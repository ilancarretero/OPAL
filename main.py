
"""
main.py

Entry point for training and evaluating prototype-based classification models.
Supports multiple backbone architectures, distance metrics, and visualization
utilities (t-SNE, training curves, prototype overlays, spatial collapse metrics).

Usage:
    python main.py [--dataset CUB-200-2011] [--base_model convnext_tiny] ...

See utils/args.py for the full list of command-line arguments.
"""

import os
import torch

from utils.args import get_args
from net.net_model import NNModel, NNModel_softmax
from utils.data import get_datasets, get_dataloaders
from train_eval.train_val import train_epoch, validate_epoch
from utils.viz import (
    interactive_tsne_plot,
    plot_metrics,
    obtain_prototypes,
    compute_spatial_prototype_collapse,
)
from utils.metrics import run_inference_and_save_metrics
from utils.misc import count_prefix_dirs, set_seeds


def main():
    # ------------------------------------------------------------------ #
    # 1. Parse command-line arguments
    # ------------------------------------------------------------------ #
    args = get_args()
    
    # ------------------------------------------------------------------ #
    # 2. Create experiment directory and log arguments
    # ------------------------------------------------------------------ #
    os.makedirs(args.output_dir, exist_ok=True)
    prx = f'{args.dataset}_{args.base_model}_{args.classifier}'
    exp = count_prefix_dirs(args.output_dir, prx)
    name_exp = f'{prx}_{exp}'
    path_exp = os.path.join(args.output_dir, name_exp)
    os.makedirs(path_exp, exist_ok=True)

    log_path = os.path.join(path_exp, 'logs')
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, 'args.txt')
    with open(log_file, "w") as f:
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
    print(f'Arguments logged in {log_file}')

    # ------------------------------------------------------------------ #
    # 3. Device and reproducibility
    # ------------------------------------------------------------------ #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seeds(seed_value=args.seed, device=device)
    
    # ------------------------------------------------------------------ #
    # 4. Instantiate model
    # ------------------------------------------------------------------ #
    model_registry = {
        'NNModel': NNModel,
        'NNModel_softmax': NNModel_softmax,
    }
    if args.model not in model_registry:
        raise ValueError(f"Unknown model: {args.model}. "
                         f"Choose from {list(model_registry.keys())}")
    model = model_registry[args.model](args)
    model.to(device)
    
    # ------------------------------------------------------------------ #
    # 5. Datasets and data loaders
    # ------------------------------------------------------------------ #
    train_ds, val_ds, test_ds = get_datasets(args)
    train_dl, val_dl, test_dl = get_dataloaders(train_ds, val_ds, test_ds, args)

    # ------------------------------------------------------------------ #
    # 6. Loss, optimizer, and training loop
    # ------------------------------------------------------------------ #
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    best_epoch = -1
    path_checkpoint = os.path.join(path_exp, 'checkpoint')
    os.makedirs(path_checkpoint, exist_ok=True)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    name_model = prx + '.pth'
    path_model = os.path.join(path_checkpoint, name_model)
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_dl, criterion, optimizer, device, alpha=0.0)
        val_loss, val_acc = validate_epoch(model, val_dl, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        print(f"Epoch [{epoch+1}/{args.epochs}]:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            if args.save_model:
                torch.save(model.state_dict(), path_model)
    print(f'Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}')

    # ------------------------------------------------------------------ #
    # 7. Post-training: load best checkpoint and generate outputs
    # ------------------------------------------------------------------ #

    # Visualization: t-SNE
    if args.save_tsne:
        path_tsne = os.path.join(path_exp, 'tsne')
        os.makedirs(path_tsne, exist_ok=True)
        interactive_tsne_plot(model, train_dl, val_dl, test_dl, device, path_tsne)
        
    # Visualization: training curves
    if args.save_curves:
        path_curves = os.path.join(path_exp, 'train_val_curves')
        os.makedirs(path_curves, exist_ok=True)
        plot_metrics(train_losses, val_losses,
                     train_accs, val_accs, path_curves)

    # Reload best checkpoint for evaluation
    if args.save_model:
        model.load_state_dict(torch.load(path_model, map_location=device,
                                         weights_only=True))

    # Quantitative metrics
    if args.save_metrics:
        path_metrics = os.path.join(path_exp, 'metrics')
        os.makedirs(path_metrics, exist_ok=True)
        run_inference_and_save_metrics(model, train_dl, 
                                       val_dl, test_dl,
                                       device, path_metrics)
        
    # Spatial Prototype Collapse (SPC) metrics
    if args.save_collapse_metrics:
        path_collapse = os.path.join(path_exp, 'metrics')
        os.makedirs(path_collapse, exist_ok=True)
        compute_spatial_prototype_collapse(
            model, args.data_dir, args.num_classes, "test", path_collapse
        )

    # Prototype visualization
    if args.save_arch_proto:
        output_dir = os.path.join(path_exp, 'prototypes_viz')
        os.makedirs(output_dir, exist_ok=True)
        obtain_prototypes(
            model, args.data_dir, args.num_classes, "test", output_dir
        )


if __name__ == "__main__":
    main()