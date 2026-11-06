import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize


embeddings_path = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2/embeddings/z_embeddings_all_events.npz"
out_dir = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2/linear_probe"

all_process_names = ["ttbar", "ggf", "dihiggs", "higgs_portal", "hidden_valley"]
n_classes = len(all_process_names)

test_fraction = 0.2
batch_size = 1024
n_epochs = 100
lr = 1e-3
weight_decay = 1e-5
random_seed = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearProbe(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_classes)

    def forward(self, z):
        return self.fc(z)


def train_linear_probe():
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(embeddings_path)
    z, labels = data["z"], data["labels"]
    latent_dim = z.shape[1]

    z_train, z_test, y_train, y_test = train_test_split(
        z, labels,
        test_size=test_fraction,
        random_state=random_seed,
        stratify=labels,
    )

    z_train_t = torch.from_numpy(z_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    z_test_t = torch.from_numpy(z_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(z_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    probe = LinearProbe(latent_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, n_epochs + 1):
        probe.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for z_batch, y_batch in train_loader:
            z_batch, y_batch = z_batch.to(device), y_batch.to(device)

            logits = probe(z_batch)
            loss = criterion(logits, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * z_batch.size(0)
            correct += (logits.argmax(dim=1) == y_batch).sum().item()
            total += z_batch.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total

        probe.eval()
        with torch.no_grad():
            test_logits = probe(z_test_t.to(device))
            test_acc = (test_logits.argmax(dim=1).cpu() == y_test_t).float().mean().item()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch}: loss {train_loss:.4f}, train acc {train_acc:.4f}, test acc {test_acc:.4f}")

    probe.eval()
    with torch.no_grad():
        test_logits = probe(z_test_t.to(device)).cpu()
        test_probs = torch.softmax(test_logits, dim=1).numpy()
        test_preds = test_logits.argmax(dim=1).numpy()

    final_acc = accuracy_score(y_test, test_preds)

    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))
    per_class_auc = {}
    for c in range(n_classes):
        per_class_auc[all_process_names[c]] = roc_auc_score(y_test_bin[:, c], test_probs[:, c])

    macro_auc = roc_auc_score(y_test_bin, test_probs, average="macro", multi_class="ovr")
    cm = confusion_matrix(y_test, test_preds)

    print(f"accuracy: {final_acc:.4f}, macro auc: {macro_auc:.4f}")
    for name, auc in per_class_auc.items():
        print(f"{name}: auc {auc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["epoch"], history["train_acc"], label="train acc")
    axes[0].plot(history["epoch"], history["test_acc"], label="test acc")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title(f"linear probe accuracy (d={latent_dim})\nmacro auc = {macro_auc:.3f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(n_classes))
    axes[1].set_yticks(range(n_classes))
    axes[1].set_xticklabels(all_process_names, rotation=45, ha="right")
    axes[1].set_yticklabels(all_process_names)
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("true")
    axes[1].set_title("confusion matrix")
    for i in range(n_classes):
        for j in range(n_classes):
            axes[1].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "linear_probe_results.png"), dpi=150)
    plt.close(fig)

    results = {
        "latent_dim": latent_dim,
        "overall_accuracy": final_acc,
        "per_class_auc": per_class_auc,
        "macro_auc": macro_auc,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(out_dir, "linear_probe_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return probe, history, results


if __name__ == "__main__":
    train_linear_probe()
