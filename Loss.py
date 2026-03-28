import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.measure import label

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        batch_size = probs.size(0)
        probs_flat = probs.view(batch_size, -1)
        targets_flat = targets.view(batch_size, -1)
        intersection = (probs_flat * targets_flat).sum(dim=1)
        dice = (2. * intersection + self.smooth) / (
            probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + self.smooth + self.eps
        )
        return 1 - dice.mean()

class BCELoss(nn.Module):
    def __init__(self, pos_weight=None):
        super(BCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='mean'
        )

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0, eps=1e-6, pos_weight=None):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth, eps=eps)
        self.bce_loss = BCELoss(pos_weight=pos_weight)

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        bce = self.bce_loss(logits, targets)
        return self.dice_weight * dice + self.bce_weight * bce

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, eps=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt + self.eps) ** self.gamma
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
        return (focal_weight * ce_loss).mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, dice_weight=0.5, focal_weight=0.5,
                 alpha=None, smooth=1.0, eps=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss = DiceLoss(smooth=smooth, eps=eps)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, eps=eps)

    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal

class TopologyIntegrityMetric:
    """
    TIM is for EVALUATION ONLY (Sec 2.c.v), NOT for training loss.
    Computes CI, FI, and TIM from binary masks.
    """
    def __init__(self, epsilon=1e-6, min_component_area=50):
        self.epsilon = epsilon
        self.min_component_area = min_component_area

    def calculate_metrics(self, mask_pred):
        if mask_pred.max() > 1:
            mask_pred = (mask_pred > 0.5).astype(np.uint8)
        labeled_mask, num_features = label(mask_pred, connectivity=2, return_num=True)
        areas = []
        for i in range(1, num_features + 1):
            area = np.sum(labeled_mask == i)
            if area >= self.min_component_area:
                areas.append(area)
        if not areas:
            return 0.0, 0, 0.0
        A_total = np.sum(areas)
        A_max = np.max(areas)
        N = len(areas)
        CI = A_max / A_total if A_total > 0 else 0.0
        FI = N
        TIM = CI * np.sqrt(1.0 / (N + self.epsilon)) * 100.0
        return CI, FI, TIM

    def calculate_improvement_rate(self, tim_baseline, tim_clahe):
        if tim_baseline == 0:
            return 0.0
        return (tim_clahe - tim_baseline) / tim_baseline * 100.0

    def identify_outliers(self, tim_baseline, tim_clahe, threshold=-20.0):
        tim_baseline = np.array(tim_baseline)
        tim_clahe = np.array(tim_clahe)
        mask = tim_baseline > 0
        ir = np.zeros_like(tim_baseline)
        ir[mask] = (tim_clahe[mask] - tim_baseline[mask]) / tim_baseline[mask] * 100.0
        outlier_indices = np.where(ir < threshold)[0]
        return outlier_indices, ir