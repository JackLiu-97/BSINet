import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import utils.distributed_utils as utils


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=0., size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, y_pred, y_true):

        logpt = F.log_softmax(y_pred)

        y_true = torch.argmax(y_true, dim=-1)
        y_true = torch.unsqueeze(y_true, dim=-1)
        logpt = logpt.gather(1, y_true)
        pt = Variable(logpt.data.exp())
        # y_true的第1维是通道维度，也即预测是哪一类的标签。gather取axis=1的索引，相当于y_true的值替换了列索引
        # 为了取到具体的值，还需要行索引，根据gather的算法，y_true中的每个值本身的位置，提供行索引。如果dim=0
        # 则本身的值替换行索引，本身的值所在的位置，提供列索引

        loss = -1 * self.alpha * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def dice_loss(y_pred, y_true, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        y_true: a tensor of shape [B*H*W, C].
        y_pred: a tensor of shape [B*H*W, C]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    prob = F.softmax(y_pred, dim=1)
    y_true = y_true.type(y_pred.type())
    # 分开前景，背景求和是为了减轻类别不平衡的情况
    intersection = torch.sum(prob * y_true, dim=0)
    cardinality = torch.sum(prob + y_true, dim=0)
    loss = (2. * intersection / (cardinality + eps)).mean()

    return 1 - loss


def hybrid_loss(y_pred, y_true):
    """Calculating the loss"""
    num_classes = y_pred.shape[1]
    focal = FocalLoss(gamma=2.0, alpha=0.2)

    y_true = y_true.permute(0, 2, 3, 1)
    y_pred = y_pred.permute(0, 2, 3, 1)

    y_pred = y_pred.reshape(-1, num_classes)
    y_true = y_true.reshape(-1, )

    valid_index = y_true < 255

    y_pred = y_pred[valid_index]
    y_true = y_true[valid_index]
    # one hot
    y_true = (torch.arange(num_classes).cuda() == torch.unsqueeze(y_true, dim=1)).long()

    bce = focal(y_pred, y_true)
    dice = dice_loss(y_pred, y_true)
    loss = 0.5 * bce + dice
    return loss


def evaluate(model, data_loader, device, num_classes, print_freq):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image1, image2, target in metric_logger.log_every(data_loader, print_freq, header):
            image1, image2, target = image1.to(device), image2.to(device), target.to(device)
            output = model(image1, image2)
            output = torch.argmax(output, dim=1)
            confmat.update(target.flatten(), output.flatten())
        confmat.reduce_from_all_processes()
    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    i = 0
    for image1, image2, target in metric_logger.log_every(data_loader, print_freq, header):
        image1, image2, target = image1.to(device), image2.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            target = torch.unsqueeze(target, 1)
            dist = model(image1, image2)
            loss = hybrid_loss(dist, target)
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        i = i + 1
        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
