import torch

def train(model, train_loader, loss_function, optimizer, device):
    model.train()
    total_loss = 0
    # Additional counters for root and non-root accuracy
    total_correct_all, total_correct_root = 0, 0
    total_all, total_root = 0, 0
    for batch_idx, (data, targets, is_root) in enumerate(train_loader):
        data, targets, is_root = data.to(device), targets.to(device), is_root.to(device)
        
        # Forward pass
        outputs = model(data)
        loss = loss_function(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_correct_all += (predicted == targets).sum().item()
        total_all += targets.size(0)

        # Calculate root accuracy
        root_mask = is_root.bool()
        if root_mask.any():
            total_correct_root += (predicted[root_mask] == targets[root_mask]).sum().item()
            total_root += root_mask.sum().item()

    train_loss = total_loss / len(train_loader)
    train_acc_all = 100. * total_correct_all / total_all
    train_acc_root = 100. * total_correct_root / total_root if total_root > 0 else 0
    return train_loss, train_acc_all, train_acc_root

def evaluate(model, val_loader, loss_function, device):
    model.eval()
    total_loss = 0
    total_correct_all, total_correct_root = 0, 0
    total_all, total_root = 0, 0
    with torch.no_grad():
        for batch_idx, (data, targets, is_root) in enumerate(val_loader):
            data, targets, is_root = data.to(device), targets.to(device), is_root.to(device)
            outputs = model(data)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_correct_all += (predicted == targets).sum().item()
            total_all += targets.size(0)

            # Calculate root accuracy
            root_mask = is_root.bool()
            if root_mask.any():
                total_correct_root += (predicted[root_mask] == targets[root_mask]).sum().item()
                total_root += root_mask.sum().item()

    val_loss = total_loss / len(val_loader)
    val_acc_all = 100. * total_correct_all / total_all
    val_acc_root = 100. * total_correct_root / total_root if total_root > 0 else 0
    return val_loss, val_acc_all, val_acc_root

