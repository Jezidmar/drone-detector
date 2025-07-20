import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)

import wandb


def log_metrics(preds, target):
    """Uploads stats to wandb; wandb generates roc, pr and conf_mat plots."""
    preds2 = np.column_stack([1.0 - preds, preds])
    wandb.log({"roc": wandb.plot.roc_curve(target, preds2, labels=None)})

    wandb.log({"pr": wandb.plot.pr_curve(target, preds2, labels=None)})
    pred_labels = (preds >= 0.5).astype(int)
    wandb.log(
        {
            "conf_mat": wandb.plot.confusion_matrix(
                y_true=target, preds=pred_labels, class_names=["no‑drone", "drone"]
            )
        }
    )


# Set style for better looking plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def plot_all_metrics(
    train_losses,
    train_accs,
    val_losses,
    val_accs,
    val_probs,
    val_labels,
    test_probs,
    test_labels,
    exp_name,
    num_classes=None,
):
    """
    Plot all metrics: training curves, ROC, and PR curves for validation and test sets

    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        val_losses: List of validation losses per epoch
        val_accs: List of validation accuracies per epoch
        val_probs: Validation probabilities from model
        val_labels: Validation true labels
        test_probs: Test probabilities from model
        test_labels: Test true labels
        exp_name: Experiment name for saving plots
        num_classes: Number of classes (if None, auto-detect from data)
    """
    # Create output directory
    output_dir = f"exp/{exp_name}/images"

    # Plot training curves
    plot_training_curves(train_losses, train_accs, val_losses, val_accs, output_dir)

    # Plot ROC and PR curves for validation set
    val_roc_auc, val_pr_auc = plot_roc_pr_curves(
        val_probs, val_labels, "validation", output_dir, num_classes
    )

    # Plot ROC and PR curves for test set
    test_roc_auc, test_pr_auc = plot_roc_pr_curves(
        test_probs, test_labels, "test", output_dir, num_classes
    )

    val_preds = val_probs >= 0.5
    # F1, prec and rec on validation set
    prec_val, rec_val, f1_val, _ = precision_recall_fscore_support(
        val_labels.numpy(), val_preds, average="binary", pos_label=1
    )

    test_preds = test_probs >= 0.5
    # F1, prec and rec on test set
    prec_test, rec_test, f1_test, _ = precision_recall_fscore_support(
        test_labels.numpy(), test_preds, average="binary", pos_label=1
    )
    # Print summary
    print(f"\n{'=' * 50}")
    print(f"RESULTS SUMMARY - {exp_name}")
    print(f"{'=' * 50}")
    print(f"Final Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Training Acc:  {train_accs[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Final Validation Acc:  {val_accs[-1]:.4f}")
    #
    print(f"Validation Precision:  {prec_val:.4f}")
    print(f"Validation Recall:   {rec_val:.4f}")
    print(f"Validation F1:   {f1_val:.4f}")
    #
    print(f"Test Precision:  {prec_test:.4f}")
    print(f"Test Recall:   {rec_test:.4f}")
    print(f"Test F1:   {f1_test:.4f}")
    #
    print(f"Plots saved to: {output_dir}")
    print(f"{'=' * 50}")

    return {
        "val_roc_auc": val_roc_auc,
        "val_pr_auc": val_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_pr_auc": test_pr_auc,
    }


def plot_training_curves(train_losses, train_accs, val_losses, val_accs, output_dir):
    """Plot training and validation loss and accuracy curves"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    epochs = range(1, len(train_losses) + 1)

    # Plot Loss curves
    ax1.plot(
        epochs,
        train_losses,
        "b-",
        label="Training Loss",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax1.plot(
        epochs,
        val_losses,
        "r-",
        label="Validation Loss",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy curves
    ax2.plot(
        epochs,
        train_accs,
        "b-",
        label="Training Accuracy",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax2.plot(
        epochs,
        val_accs,
        "r-",
        label="Validation Accuracy",
        linewidth=2,
        marker="s",
        markersize=4,
    )
    ax2.set_title("Training and Validation Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_curves.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training curves saved to {output_dir}/training_curves.png")


def plot_roc_pr_curves(probs, labels, dataset_name, output_dir, num_classes=None):
    """Plot ROC and PR curves for a dataset"""

    # Convert to numpy if torch tensors
    if torch.is_tensor(probs):
        probs = probs.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    # Auto-detect number of classes if not provided
    if num_classes is None:
        num_classes = len(np.unique(labels))

    return plot_binary_roc_pr(probs, labels, dataset_name, output_dir)


def plot_binary_roc_pr(probs, labels, dataset_name, output_dir):
    """Plot ROC and PR curves for binary classification"""

    y_scores = probs

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, y_scores)
    roc_auc = auc(fpr, tpr)

    # PR Curve
    precision, recall, _ = precision_recall_curve(labels, y_scores)
    pr_auc = average_precision_score(labels, y_scores)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # ROC Curve
    ax1.plot(
        fpr, tpr, color="darkorange", lw=3, label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    ax1.plot(
        [0, 1], [0, 1], color="navy", lw=2, linestyle="--", alpha=0.8, label="Random"
    )
    ax1.fill_between(fpr, tpr, alpha=0.2, color="darkorange")
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel("False Positive Rate", fontsize=12)
    ax1.set_ylabel("True Positive Rate", fontsize=12)
    ax1.set_title(
        f"ROC Curve - {dataset_name.title()} Set", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # PR Curve
    ax2.plot(
        recall,
        precision,
        color="darkgreen",
        lw=3,
        label=f"PR curve (AP = {pr_auc:.3f})",
    )
    ax2.fill_between(recall, precision, alpha=0.2, color="darkgreen")
    ax2.axhline(
        y=np.mean(labels),
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Random (AP = {np.mean(labels):.3f})",
    )
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel("Recall", fontsize=12)
    ax2.set_ylabel("Precision", fontsize=12)
    ax2.set_title(
        f"Precision-Recall Curve - {dataset_name.title()} Set",
        fontsize=14,
        fontweight="bold",
    )
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        f"{output_dir}/roc_pr_curves_{dataset_name}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"ROC and PR curves saved to {output_dir}/roc_pr_curves_{dataset_name}.png")
    return roc_auc, pr_auc
