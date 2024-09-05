import torch
import torch.nn as nn
import torch.nn.functional as F

class ECSTFL(nn.Module):
    def __init__(self):
        super(ECSTFL, self).__init__()
        pass

    def forward(self, features, labels):
        # calcalting the distance matrix between all sample pairs
        dist_matrix = torch.cdist(features, features, p=1)  # L1 norm

        # acquring masks for the same labels and the different labels
        labels = labels.unsqueeze(1)
        mask_positive = torch.eq(labels, labels.T).float()
        mask_negative = torch.ne(labels, labels.T).float()

        # attractive loss to narrows the distance between the samples with same labels
        positive_loss = dist_matrix * mask_positive
        positive_count = mask_positive.sum(dim=1) - 1  # minus itself
        attractive_loss = positive_loss.sum(dim=1) / (positive_count + 1 + 1e-6)  # avoiding dividing zero

        # repulsive loss to extend the distance between the samples with different labels
        negative_loss = 1.0 / (dist_matrix + 1e-6) * mask_negative
        negative_count = mask_negative.sum(dim=1)
        repulsive_loss = negative_loss.sum(dim=1) / (negative_count + 1 + 1e-6)  # avoiding dividing zero

        total_loss = attractive_loss * repulsive_loss / features.size(0)  # averging to each sample
        total_loss = total_loss.mean()

        return total_loss




if __name__ == "__main__":
    # make the feature matrix and the corresponding labels
    features = torch.randn(10, 128)  # 10 samples (dim_{fea}=128)
    labels = torch.tensor([1, 2, 1, 3, 2, 2, 3, 1, 3, 2])  # labels for 10 samples

    ECSTFL_layer = ECSTFL()
    loss = ECSTFL_layer(features, labels)

    print('Calculated Loss:', loss)