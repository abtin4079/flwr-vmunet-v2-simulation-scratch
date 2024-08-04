import torch


def test_v2(net, testloader, device: str, criterion):
    """Validate the network on the entire test set."""
    correct, loss, iou, dsc, f1, recall, precision = 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            iou += intersection_over_union(predicted, labels).sum().item()
            dsc += dice_similarity_coefficient(predicted, labels).sum().item()
            f10 , recall0, precision0 = f1_score(predicted, labels).sum().item()
            f1 += f10
            recall += recall0
            precision += precision0

    accuracy = correct / len(testloader.dataset)
    IOU = iou / len(testloader.dataset)
    DSC = dsc / len(testloader.dataset)
    F1 = f1 / len(testloader.dataset) 
    RECALL = recall / len(testloader.dataset)
    PERCISION = precision / len(testloader.dataset)

    metrics = []

    metrics.append(IOU)
    metrics.append(DSC)
    metrics.append(F1)
    metrics.append(RECALL)
    metrics.append(PERCISION)


    return loss, metrics


################################ metrics ######################

import torch
from utils import dice_loss

def intersection_over_union(pred, target):
    pred = torch.sigmoid(pred)
    
    pred_binary = (pred > 0.5).float()  # Convert probabilities to binary predictions
    target_binary = target.float()

    intersection = torch.sum(pred_binary * target_binary)
    union = torch.sum(pred_binary) + torch.sum(target_binary) - intersection

    epsilon = 1e-5
    iou = intersection / (union + epsilon)

    return iou.item()


def dice_similarity_coefficient(pred, target):
    dice_score = dice_loss(pred, target)
    DSC = 1 - dice_score
    return DSC.item()
    

def accuracy(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()  # Convert probabilities to binary predictions
    target_binary = target.float()

    correct = torch.sum((pred_binary == target_binary).float())
    total = target.numel()

    accuracy = correct / total

    return accuracy.item()


def f1_score(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    
    pred_binary = (pred > threshold).float()  # Convert probabilities to binary predictions
    target_binary = target.float()

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
    y_true = torch.tensor([[0, 1, 1], [1, 0, 1]])
    y_pred = torch.tensor([[0, 1, 0], [1, 1, 1]])

    print("Intersection over Union:", intersection_over_union(y_true, y_pred))
    print("Dice Similarity Coefficient:", dice_similarity_coefficient(y_true, y_pred))
    # print("F1 Score:", f1_score(y_true.flatten(), y_pred.flatten()))