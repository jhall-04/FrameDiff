from stage5_model import FrameDiffModel
from stage4_dataloader import make_dataloaders
import torch
import yaml

train_loader, val_loader, test_loader = make_dataloaders("configs/stage_6_config.yaml")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FrameDiffModel().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(x_batch).squeeze()
        loss = criterion(outputs, y_batch.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x_batch.size(0)
    return train_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).squeeze()
            loss = criterion(outputs, y_batch.float())
            val_loss += loss.item() * x_batch.size(0)
            preds = (outputs > 0).float()
            correct += (preds == y_batch).sum().item()
    return val_loss / len(loader.dataset), correct / len(loader.dataset)

best_val_loss = float('inf')

print("Starting training...")
for epoch in range(15):
    print(f"Epoch {epoch+1}/15")
    print("Training...")
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    print("Evaluating...")
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_framediff_model.pt")
    
    scheduler.step()


model.load_state_dict(torch.load("best_framediff_model.pt"))
model.eval()
with torch.no_grad():
    output = model(test_loader.dataset[0][0].unsqueeze(0).to(device))
    print(f"Test output for first sample: {output.item():.4f}")