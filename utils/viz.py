"""viz.py

Visualization utilities: interactive t-SNE plots of model embeddings,
training / validation metric curves, prototype overlay reports, and
Spatial Prototype Collapse (SPC) computation.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw

def extract_embeddings(
        model: torch.nn.Module,
        loader: DataLoader,
        group_label: str,
        device: torch.device
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract embeddings, labels, and group tags from a data loader.
        """
        embeddings_list, labels_list, groups_list = [], [], []

        for images, labels in loader:
            # Move inputs to the correct device
            images = images.to(device)
            # Obtain embeddings from model
            emb = model.get_embedding(images)
            # Collect results on CPU
            embeddings_list.append(emb.cpu().numpy())
            labels_list.append(labels.numpy())
            # Tag each sample with its dataset group
            groups_list.append(np.full(labels.shape, group_label))

        # Concatenate lists into arrays
        embeddings = np.concatenate(embeddings_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        groups = np.concatenate(groups_list, axis=0)
        return embeddings, labels, groups

def interactive_tsne_plot(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    tsne_dir: str
) -> None:
    """
    Generate and save an interactive t-SNE plot of embeddings for train/val/test datasets.

    Args:
        model: PyTorch model with a `get_embedding` method.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        test_loader: DataLoader for test data.
        device: Device to run the model on (e.g., 'cuda' or 'cpu').
        tsne_dir: Directory path where the HTML plot should be saved.
    """
    # Switch model to evaluation mode
    model.eval()

    # Extract embeddings for each split
    train_emb, train_labels, train_group = extract_embeddings(model, train_loader, "Train", device)
    val_emb, val_labels, val_group       = extract_embeddings(model, val_loader,   "Validation", device)
    test_emb, test_labels, test_group    = extract_embeddings(model, test_loader,  "Test", device)

    # Combine all embeddings, labels, and group tags
    all_embeddings = np.vstack([train_emb, val_emb, test_emb])
    all_labels     = np.concatenate([train_labels, val_labels, test_labels])
    all_groups     = np.concatenate([train_group, val_group, test_group])

    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        "TSNE1": np.zeros(all_embeddings.shape[0]),
        "TSNE2": np.zeros(all_embeddings.shape[0]),
        "Class": all_labels.astype(str),
        "Dataset": all_groups
    })

    # Run t-SNE on combined embeddings
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(all_embeddings)
    df["TSNE1"], df["TSNE2"] = tsne_results[:, 0], tsne_results[:, 1]

    # Create interactive scatter plot
    fig = px.scatter(
        df,
        x="TSNE1",
        y="TSNE2",
        color="Class",
        symbol="Dataset",
        title="Interactive t-SNE of Model Embeddings",
        labels={"TSNE1": "Dimension 1", "TSNE2": "Dimension 2"},
        hover_data=["Class", "Dataset"]
    )

    # Ensure output directory exists
    os.makedirs(tsne_dir, exist_ok=True)
    output_path = os.path.join(tsne_dir, "interactive_tsne.html")
    # Save plot to HTML file
    fig.write_html(output_path)
    print(f"Interactive t-SNE plot saved to {output_path}")

def plot_metrics(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    metrics_dir: str
) -> None:
    """
    Plot and save training and validation loss/accuracy curves.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses:   List of validation loss values per epoch.
        train_accs:   List of training accuracy percentages per epoch.
        val_accs:     List of validation accuracy percentages per epoch.
        metrics_dir:  Directory path where the PNG plot should be saved.
    """
    # Generate epoch index array
    epochs = np.arange(1, len(train_losses) + 1)

    # Ensure output directory exists
    os.makedirs(metrics_dir, exist_ok=True)

    # Create a two-panel plot: loss and accuracy
    plt.figure(figsize=(12, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses,  marker='o', label='Train Loss')
    plt.plot(epochs, val_losses,    marker='o', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs,  marker='o', label='Train Accuracy')
    plt.plot(epochs, val_accs,    marker='o', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    output_path = os.path.join(metrics_dir, "metrics_plot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Metrics plot saved to {output_path}")


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    """
    Extension of ImageFolder to return image paths
    """
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path
    
def get_dataset(root, split, num_classes):
    path = os.path.join(root, split)
    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = ImageFolderWithPaths(path, transform)
    if num_classes:
        idxs = [i for i,(_,lbl,_) in enumerate(ds) if lbl < num_classes]
        ds = Subset(ds, idxs)
    names = ds.dataset.classes if isinstance(ds, Subset) else ds.classes
    return ds, names

def run_inference(model, loader, device):
    protos = model.classifier.prototypes.cpu().numpy()
    emb_list, dist_list, lbl_list, pred_list, path_list = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            emb = model.get_embedding(imgs)
            emb_np = emb.cpu().numpy()
            preds = logits.argmax(1).cpu().numpy()
            true = labels.numpy()
            dists = np.linalg.norm(emb_np - protos[true], axis=1)
            emb_list.append(emb_np)
            dist_list.extend(dists.tolist())
            lbl_list.extend(true.tolist())
            pred_list.extend(preds.tolist())
            path_list.extend(paths)
    embeddings = np.vstack(emb_list)
    return embeddings, np.array(dist_list), np.array(lbl_list), np.array(pred_list), path_list


def save_class_reports(
    cls, class_names, idxs, labels, preds, dists, paths, embeddings,
    prototypes, colors, model, device, output_base
):
    """
    Saves Excel, sorted images with proto overlays, and sub-proto crops.

    Saves both native-resolution crops (no resize) and 224x224 thumbnails (LANCZOS).
    """
    model.eval()
    with torch.no_grad():
        proto_vec = prototypes[cls]
        ch_idxs = np.where(proto_vec != 0)[0][:len(colors)]
        # sort indices by increasing distance for ranking
        sorted_asc = idxs[np.argsort(dists[idxs])]
        # prepare Excel rows
        rows = []
        for rank, idx in enumerate(sorted_asc):
            row = {
                'rank': rank,
                'image': os.path.basename(paths[idx]),
                'true_class_idx': cls,
                'true_class_name': class_names[cls],
                'pred_idx': int(preds[idx]),
                'pred_name': class_names[int(preds[idx])],
                'correct': preds[idx] == cls,
                'distance': float(dists[idx])
            }
            emb = embeddings[idx]
            for pi, ch in enumerate(ch_idxs):
                col = f'P{pi+1}_{colors[pi]}'
                row[col] = float(abs(proto_vec[ch] - emb[ch]))
            rows.append(row)
        df = pd.DataFrame(rows)
        excel_path = os.path.join(output_base, 'report.xlsx')
        df.to_excel(excel_path, index=False)

        img_dir = os.path.join(output_base, 'images')
        crop_base = os.path.join(output_base, 'prototypes')
        os.makedirs(img_dir, exist_ok=True)
        for pi in range(len(ch_idxs)):
            os.makedirs(os.path.join(crop_base, f'P{pi+1}_{colors[pi]}'), exist_ok=True)
            os.makedirs(os.path.join(crop_base, f'P{pi+1}_{colors[pi]}', 'native'), exist_ok=True)
            os.makedirs(os.path.join(crop_base, f'P{pi+1}_{colors[pi]}', 'thumbs'), exist_ok=True)

        # overlay rectangles: farthest first (descending distance)
        sorted_desc = sorted_asc[::-1]
        rect_width = 3  # uniform border thickness in pixels on the display image
        for rank, idx in enumerate(sorted_desc):
            # load original image (native resolution)
            orig_img = Image.open(paths[idx]).convert('RGB')
            # create a display image resized to 224x224 for model fmap and overlays
            disp_img = orig_img.resize((224, 224))
            draw = ImageDraw.Draw(disp_img)

            # get feature map from model for the display image
            fmap = model.get_last_conv(T.ToTensor()(disp_img).unsqueeze(0).to(device))[0].cpu().numpy()
            H, W = fmap.shape[1:]
            ph, pw = 224 / H, 224 / W

            # scale factors to map display coords back to native resolution
            scale_x = orig_img.width / 224.0
            scale_y = orig_img.height / 224.0

            for pi, ch in enumerate(ch_idxs):
                fm = fmap[ch]
                y, x = np.unravel_index(np.argmax(fm), fm.shape)
                x0_f, y0_f = x * pw, y * ph
                x1_f, y1_f = x0_f + pw, y0_f + ph

                # integer coordinates for uniform border rendering on display image
                x0_i, y0_i = int(round(x0_f)), int(round(y0_f))
                x1_i, y1_i = int(round(x1_f)), int(round(y1_f))
                x0_i, y0_i = max(0, x0_i), max(0, y0_i)
                x1_i, y1_i = min(224, x1_i), min(224, y1_i)

                # draw rectangle on display image
                draw.rectangle([x0_i, y0_i, x1_i, y1_i], outline=colors[pi], width=rect_width)

                # compute native-resolution bbox and clip
                nx0 = int(round(x0_f * scale_x))
                ny0 = int(round(y0_f * scale_y))
                nx1 = int(round(x1_f * scale_x))
                ny1 = int(round(y1_f * scale_y))
                nx0, ny0 = max(0, nx0), max(0, ny0)
                nx1, ny1 = min(orig_img.width, nx1), min(orig_img.height, ny1)

                # native crop (no resize) — for scientific reproducibility / supplement
                native_crop = orig_img.crop((nx0, ny0, nx1, ny1))
                native_name = f"{rank}_P{pi+1}_native_{os.path.basename(paths[idx])}"
                native_path = os.path.join(crop_base, f'P{pi+1}_{colors[pi]}', 'native', native_name)
                native_crop.save(native_path, dpi=(300, 300))

                # thumbnail crop (normalized to 224x224) — for figures/comparisons
                thumb = native_crop.resize((224, 224), resample=Image.LANCZOS)
                thumb_name = f"{rank}_P{pi+1}_thumb_{os.path.basename(paths[idx])}"
                thumb_path = os.path.join(crop_base, f'P{pi+1}_{colors[pi]}', 'thumbs', thumb_name)
                thumb.save(thumb_path, dpi=(300, 300))

            # save overlaid full display image with correct rank prefix
            out_name = f"{rank}_{os.path.basename(paths[idx])}"
            disp_img.save(os.path.join(img_dir, out_name))

        # per-image bar charts
        for rank, idx in enumerate(sorted_asc):
            emb = embeddings[idx]
            plt.figure()
            vals = [emb[ch] for ch in ch_idxs]
            plt.bar(
                [f'P{i+1}' for i in range(len(ch_idxs))],
                vals,
                color=[colors[i] for i in range(len(ch_idxs))]
            )
            plt.xlabel('Sub-prototypes')
            plt.ylabel('Embedding value')
            plt.title(f'Class {cls} Image {rank}')
            plt.tight_layout()
            # base filename without extension
            base_fn = os.path.splitext(os.path.basename(paths[idx]))[0]
            bar_name = f"{rank}_{base_fn}.png"
            plt.savefig(os.path.join(img_dir, bar_name))
            plt.close()
            
def obtain_prototypes(model, data_dir, num_classes, split, output_dir):
    colors = [
        "#8DD3C7",  # mint
        "#FFFFB3",  # pale yellow
        "#BEBADA",  # lavender
        "#FB8072",  # coral
        "#80B1D3",  # soft blue
        "#FDB462",  # pastel orange
        "#B3DE69",  # light green
        "#FCCDE5",  # pastel pink
        "#BC80BD",  # purple
        "#A6CEE3",  # sky blue 
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    
    ds, class_names = get_dataset(data_dir, split, num_classes)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    embeddings, dists, labels, preds, paths = run_inference(model, loader, device)
    for cls in np.unique(labels):
        idxs = np.where(labels == cls)[0]
        base = os.path.join(
            output_dir, 'prototypes',
            split, f'class_{cls}'
        )
        os.makedirs(base, exist_ok=True)
        save_class_reports(
            cls,
            class_names,
            idxs,
            labels,
            preds,
            dists,
            paths,
            embeddings,
            model.classifier.prototypes.cpu().numpy(),
            colors,
            model,
            device,
            base
        )


def compute_spatial_prototype_collapse(model, data_dir, num_classes, split, output_dir):
    """
    Compute Spatial Prototype Collapse (SPC) metric for each image in a dataset split.

    For each image, K sub-prototypes (non-zero channels of the class prototype)
    each select a spatial position via argmax on the feature map.
        SPC(x) = 1 - (num_unique_positions - 1) / (K - 1)
    SPC = 0: all sub-prototypes attend to different spatial positions (no collapse)
    SPC = 1: all sub-prototypes attend to the same position (total collapse)

    Saves:
        - spatial_collapse_metrics.csv: summary statistics (mean, std, median, min, max)
        - spatial_collapse_per_image.csv: per-image SPC values for later analysis
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    ds, class_names = get_dataset(data_dir, split, num_classes)
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    prototypes = model.classifier.prototypes.cpu().numpy()

    rows = []
    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            fmaps = model.get_last_conv(imgs).cpu().numpy()  # [B, C, H, W]

            for i in range(imgs.size(0)):
                fmap = fmaps[i]  # [C, H, W]
                cls = labels[i].item()
                proto_vec = prototypes[cls]
                ch_idxs = np.where(proto_vec != 0)[0]
                K = len(ch_idxs)

                if K <= 1:
                    spc = 0.0
                    unique = K
                else:
                    positions = set()
                    for ch in ch_idxs:
                        fm = fmap[ch]
                        pos = np.unravel_index(np.argmax(fm), fm.shape)
                        positions.add(pos)
                    unique = len(positions)
                    spc = 1.0 - (unique - 1) / (K - 1)

                rows.append({
                    'image': os.path.basename(paths[i]),
                    'class_idx': cls,
                    'class_name': class_names[cls],
                    'num_subprototypes': K,
                    'unique_positions': unique,
                    'SPC': round(spc, 6)
                })

    df_per_image = pd.DataFrame(rows)

    spc_values = df_per_image['SPC'].values
    summary = {
        'SPC_mean': [round(float(np.mean(spc_values)), 6)],
        'SPC_std': [round(float(np.std(spc_values)), 6)],
        'SPC_median': [round(float(np.median(spc_values)), 6)],
        'SPC_min': [round(float(np.min(spc_values)), 6)],
        'SPC_max': [round(float(np.max(spc_values)), 6)],
        'num_images': [len(spc_values)]
    }
    df_summary = pd.DataFrame(summary)

    os.makedirs(output_dir, exist_ok=True)
    per_image_path = os.path.join(output_dir, 'spatial_collapse_per_image.csv')
    summary_path = os.path.join(output_dir, 'spatial_collapse_metrics.csv')
    df_per_image.to_csv(per_image_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print(f"Spatial Prototype Collapse metrics saved to: {output_dir}")
    print(df_summary.to_string(index=False))

