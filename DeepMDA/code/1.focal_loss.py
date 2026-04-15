import torch
import torch.nn.functional as F


def focal_loss(logit, target, alpha=0.25, gamma=2.0, reduction='mean'):
    """
    鲁棒版二分类Focal Loss
    输入兼容：
    - logit: [B] / [B,1] / [B,N]（自动压缩到[B]）
    - target: [B] / [B,1]（自动压缩到[B]）
    """
    # 1. 统一设备
    target = target.to(logit.device)

    # 2. 压缩维度到一维
    logit = logit.view(-1)  # [B,N] → [B*N]（若N=256，B=32则变成8192，需先确认模型输出）
    target = target.view(-1).float()  # 转float避免BCE报错

    # 3. 校验长度匹配（关键：避免维度不匹配）
    if len(logit) != len(target):
        # 若logit长度是target的整数倍（比如logit=32*256，target=32），则reshape
        if len(logit) % len(target) == 0:
            logit = logit.view(len(target), -1).mean(dim=1)  # [32*256] → [32,256] → [32]
        else:
            raise ValueError(f"logit长度{len(logit)}和target长度{len(target)}不匹配，且无法整除！")

    # 4. 计算Focal Loss
    prob = torch.sigmoid(logit)
    pt = torch.where(target == 1, prob, 1 - prob)  # 更简洁的pt计算方式
    focal_weight = (1 - pt) ** gamma
    bce_loss = F.binary_cross_entropy_with_logits(
        logit, target,
        pos_weight=torch.tensor([alpha / (1 - alpha)], device=logit.device) if alpha < 1 else None,
        reduction='none'
    )
    focal_loss = focal_weight * bce_loss

    # 5. 按reduction返回
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss