import torch
import torch.nn as nn
import numpy as np


class SegmentationCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(SegmentationCrossEntropyLoss, self).__init__()

    def __len__(self):
        """ length of the components of loss to display """
        return 1

    def forward(self, output, labels):
        # out = output[0]
        out = output
        true_masks = labels
        masks_probs = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        masks_probs = masks_probs.reshape(masks_probs.shape[0] * masks_probs.shape[1] * masks_probs.shape[2],
                                          masks_probs.shape[3])  # (B * H * W, C)
        true_masks = true_masks.view(-1)  # (B * T * H * W)
        loss_s = nn.CrossEntropyLoss(reduction='none')(masks_probs, true_masks)
        loss_s = torch.mean(loss_s)
        return loss_s, masks_probs


class SegmentationDiceCoefficient(nn.Module):
    def __init__(self):
        super(SegmentationDiceCoefficient, self).__init__()

    def forward(self, true_masks, out):

        n_classes = out.shape[1]
        masks_probs = out.permute(0, 2, 3, 1)  # (B, H, W, C)
        masks_probs = masks_probs.reshape(masks_probs.shape[0] * masks_probs.shape[1] * masks_probs.shape[2],
                                          masks_probs.shape[3])  # (B * H * W, C)
        _, masks_pred = torch.max(masks_probs, 1)

        dice = np.zeros(n_classes)
        dice_tp = np.zeros(n_classes)
        dice_div = np.zeros(n_classes)
        for c in range(n_classes):
            dice_tp[c] += ((masks_pred == c) & (true_masks.view(-1) == c)).sum().item()
            dice_div[c] += ((masks_pred == c).sum().item() + (true_masks.view(-1) == c).sum().item())
            dice[c] = 2 * dice_tp[c] / dice_div[c]

        return dice[:]  # omit the background channel
