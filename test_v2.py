import torch

def test_v2(net, testloader, device: str, criterion):
    """Validate the network on the entire test set."""
    correct, total_loss, iou, dsc, f1, recall, precision = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            img, msk = data
            images, labels = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            outputs = net(images)  
            loss = criterion(outputs, labels).item()
            total_loss += loss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            iou += intersection_over_union(predicted, labels)
            dsc += dice_similarity_coefficient(predicted, labels)
            f10, recall0, precision0 = f1_score(predicted, labels)
            f1 += f10
            recall += recall0
            precision += precision0

    accuracy = correct / len(testloader.dataset)
    IOU = iou / len(testloader)
    DSC = dsc / len(testloader)
    F1 = f1 / len(testloader) 
    RECALL = recall / len(testloader)
    PRECISION = precision / len(testloader)

    metrics = [IOU, DSC, F1, RECALL, PRECISION]

    return total_loss, metrics

################################ Metrics ######################

def intersection_over_union(pred, target):
    pred = pred.double()
    target = target.double()
    
    pred_binary = (pred > 0.5).double()
    target_binary = target

    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary) - intersection

    epsilon = 1e-5
    iou = intersection / (union + epsilon)

    return iou.item()

def dice_similarity_coefficient(pred, target):
    pred = pred.double()
    target = target.double()
    
    pred_binary = (pred > 0.5).double()
    target_binary = target

    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary)

    epsilon = 1e-5
    dice_score = 2 * intersection / (union + epsilon)
    DSC = dice_score

    return DSC.item()

def f1_score(pred, target, threshold=0.5):
    pred = pred.double()
    target = target.double()
    
    pred_binary = (pred > threshold).double()
    target_binary = target

    tp = torch.sum(pred_binary * target_binary)
    fp = torch.sum((1 - target_binary) * pred_binary)
    fn = torch.sum(target_binary * (1 - pred_binary))

    epsilon = 1e-5
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return f1.item(), recall.item(), precision.item()

if __name__ == '__main__':
    # Example usage:
    y_true = torch.tensor([[0, 1, 1], [1, 0, 1]], dtype=torch.float64)
    y_pred = torch.tensor([[0, 1, 0], [1, 1, 1]], dtype=torch.float64)

    print("Intersection over Union:", intersection_over_union(y_true, y_pred))
    print("Dice Similarity Coefficient:", dice_similarity_coefficient(y_true, y_pred))
    print("F1 Score:", f1_score(y_true.flatten(), y_pred.flatten()))
