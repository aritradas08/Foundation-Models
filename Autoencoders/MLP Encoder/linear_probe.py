import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import label_binarize

EMBEDDINGS_PATH = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2/embeddings/z_embeddings_all_events.npz"
OUT_DIR = "/global/cfs/cdirs/m4958/usr/aritra08/autoencoders/mlp/results/sm_only_v3_run2/linear_probe"

ALL_PROCESS_NAMES = ["ttbar", "ggf", "dihiggs", "higgs_portal", "hidden_valley"]
N_CLASSES = len(ALL_PROCESS_NAMES)

TEST_FRACTION = 0.2
BATCH_SIZE = 1024
N_EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LinearProbe(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_classes)

    def forward(self, z):
        return self.fc(z)

def train_linear_probe():
    os.makedirs(OUT_DIR, exist_ok=True)

    data = np.load(EMBEDDINGS_PATH)
    z, labels = data["z"], data["labels"]
    print(f"z: {z.shape}, labels: {labels.shape}")
    latent_dim = z.shape[1]

    for label_val, name in enumerate(ALL_PROCESS_NAMES):
        n = (labels == label_val).sum()
        print(f"label={label_val} ({name}): {n} events")

    z_train, z_test, y_train, y_test = train_test_split(
        z, labels,
        test_size=TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    print(f"Train: {z_train.shape[0]} events, Test: {z_test.shape[0]} events")

    z_train_t = torch.from_numpy(z_train).float()
    y_train_t = torch.from_numpy(y_train).long()
    z_test_t = torch.from_numpy(z_test).float()
    y_test_t = torch.from_numpy(y_test).long()

    train_dataset = torch.utils.data.TensorDataset(z_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    probe = LinearProbe(latent_dim, N_CLASSES).to(DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, N_EPOCHS + 1):
        probe.train()
        epoch_loss, correct, total = 0.0, 0, 0

        for z_batch, y_batch in train_loader:
            z_batch, y_batch = z_batch.to(DEVICE), y_batch.to(DEVICE)

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
            test_logits = probe(z_test_t.to(DEVICE))
            test_acc = (test_logits.argmax(dim=1).cpu() == y_test_t).float().mean().item()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{N_EPOCHS}; loss={train_loss:.4f}; train_acc={train_acc:.4f}; test_acc={test_acc:.4f}")
                  
    probe.eval()
    with torch.no_grad():
        test_logits = probe(z_test_t.to(DEVICE)).cpu().numpy()
        test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        test_preds = test_logits.argmax(axis=1)

    

    final_acc = accuracy_score(y_test, test_preds)
    print(f"\nOverall accuracy: {final_acc:.4f}")

    y_test_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))
    print("\nPer-class AUC:")
    per_class_auc = {}
    for c in range(N_CLASSES):
        auc = roc_auc_score(y_test_bin[:, c], test_probs[:, c])
        per_class_auc[ALL_PROCESS_NAMES[c]] = auc
        print(f"{ALL_PROCESS_NAMES[c]:15s}: AUC = {auc:.4f}")

    macro_auc = roc_auc_score(y_test_bin, test_probs, average="macro", multi_class="ovr")
    print(f"\nMacro-average AUC: {macro_auc:.4f}")
    
    cm = confusion_matrix(y_test, test_preds)
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "             " + "".join(f"{n[:8]:>10s}" for n in ALL_PROCESS_NAMES)
    print(header)
    for i, name in enumerate(ALL_PROCESS_NAMES):
        row = "".join(f"{cm[i,j]:>10d}" for j in range(N_CLASSES))
        print(f"  {name:10s} {row}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history["epoch"], history["train_acc"], label="train acc")
    axes[0].plot(history["epoch"], history["test_acc"], label="test acc")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("accuracy")
    axes[0].set_title(f"Linear probe accuracy (D={latent_dim})\nmacro AUC = {macro_auc:.3f}")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set_xticks(range(N_CLASSES))
    axes[1].set_yticks(range(N_CLASSES))
    axes[1].set_xticklabels(ALL_PROCESS_NAMES, rotation=45, ha="right")
    axes[1].set_yticklabels(ALL_PROCESS_NAMES)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Confusion matrix")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            axes[1].text(j, i, cm[i, j], ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=axes[1], fraction=0.046)

    fig.tight_layout()
    plot_path = os.path.join(OUT_DIR, "linear_probe_results.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {plot_path}")

    results = {
        "latent_dim": latent_dim,
        "overall_accuracy": final_acc,
        "per_class_auc": per_class_auc,
        "macro_auc": macro_auc,
        "confusion_matrix": cm.tolist(),
    }
    results_path = os.path.join(OUT_DIR, "linear_probe_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")

    return probe, history, results


if __name__ == "__main__":
    train_linear_probe()
