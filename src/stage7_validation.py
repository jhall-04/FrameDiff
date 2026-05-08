from stage5_model import FrameDiffModel
from stage4_dataloader import make_dataloaders
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch

train_loader, val_loader, test_loader = make_dataloaders("configs/stage_6_config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FrameDiffModel().to(device)
model.load_state_dict(torch.load("/home/whall/FrameDiff/models/best_framediff_model.pt"))
model.eval()

criterion = torch.nn.BCEWithLogitsLoss()

# Evaluation loop + detailed metrics
def evaluate(model, loader, criterion):
    model.eval()
    val_loss, correct = 0.0, 0
    buckets = {}
    counts = {}
    i = 0
    outputs_list, y_list = [], []
    with torch.no_grad():
        for x_batch, k_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).squeeze()
            outputs_list.append(outputs.cpu())
            y_list.append(y_batch.cpu())
            loss = criterion(outputs, y_batch.float())
            val_loss += loss.item() * x_batch.size(0)
            preds = (outputs > 0).float()
            for pred, k, true in zip(preds.cpu().numpy(), k_batch.cpu().numpy(), y_batch.cpu().numpy()):
                if pred == true:
                    correct += 1
                    buckets[k] = buckets.get(k, 0) + 1
                counts[k] = counts.get(k, 0) + 1
            i += x_batch.size(0)
            if i % 1000 == 0:
                print(f"Evaluated {i} samples...")
    return val_loss / len(loader.dataset), correct / len(loader.dataset), buckets, counts, torch.cat(outputs_list), torch.cat(y_list)


if __name__ == "__main__":
    # Run detailed evaluation on validation set, including per-k accuracy and ROC curve, then do the same on the test set.
    print("Evaluating on validation set...")
    val_loss, val_acc, val_buckets, val_counts, val_outputs, val_targets = evaluate(model, val_loader, criterion)
    print(f"Validation Loss={val_loss:.4f}, Validation Accuracy={val_acc:.4f}")
    print(f"Validation Buckets: {val_buckets}")
    print(f"Validation Counts: {val_counts}")
    for k in sorted(val_counts.keys()):
        acc = val_buckets.get(k, 0) / val_counts[k]
        print(f"  k={k}: {acc:.4f} ({val_buckets.get(k, 0)}/{val_counts[k]})")
    probs = torch.sigmoid(val_outputs).cpu().numpy()
    fpr, tpr, thresholds = roc_curve(val_targets.cpu().numpy(), probs)
    val_auc = roc_auc_score(val_targets, probs)
    print(f"Validation AUC={val_auc:.4f}")
    positives = (val_outputs > 0).float().cpu().numpy()
    negatives = (val_outputs <= 0).float().cpu().numpy()
    tp = ((positives == 1) & (val_targets.cpu().numpy() == 1)).sum()
    tn = ((negatives == 1) & (val_targets.cpu().numpy() == 0)).sum()
    fp = ((positives == 1) & (val_targets.cpu().numpy() == 0)).sum()
    fn = ((negatives == 1) & (val_targets.cpu().numpy() == 1)).sum()
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}")

    plt.plot(fpr, tpr, label=f"Val AUC={val_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Validation ROC Curve")
    plt.legend()
    plt.savefig("validation_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.clf()

    test_loss, test_acc, test_buckets, test_counts, test_outputs, test_targets = evaluate(model, test_loader, criterion)
    print(f"Test Loss={test_loss:.4f}, Test Accuracy={test_acc:.4f}")
    print(f"Test Buckets: {test_buckets}")
    print(f"Test Counts: {test_counts}")
    for k in sorted(test_counts.keys()):
        acc = test_buckets.get(k, 0) / test_counts[k]
        print(f"  k={k}: {acc:.4f} ({test_buckets.get(k, 0)}/{test_counts[k]})")
    probs = torch.sigmoid(test_outputs).cpu().numpy()
    fpr, tpr, thresholds = roc_curve(test_targets.cpu().numpy(), probs)
    test_auc = roc_auc_score(test_targets, probs)
    print(f"Test AUC={test_auc:.4f}")
    positives = (test_outputs > 0).float().cpu().numpy()
    negatives = (test_outputs <= 0).float().cpu().numpy()
    tp = ((positives == 1) & (test_targets.cpu().numpy() == 1)).sum()
    tn = ((negatives == 1) & (test_targets.cpu().numpy() == 0)).sum()
    fp = ((positives == 1) & (test_targets.cpu().numpy() == 0)).sum()
    fn = ((negatives == 1) & (test_targets.cpu().numpy() == 1)).sum()
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}")

    plt.plot(fpr, tpr, label=f"Test AUC={test_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Test ROC Curve")
    plt.legend()
    plt.savefig("test_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.clf()
