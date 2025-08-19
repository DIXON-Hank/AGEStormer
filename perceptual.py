
# --- Imports --- #
import torch
import torchvision
import torch.nn.functional as F
from kornia.filters import sobel

# --- Perceptual loss network  --- #
class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def output_features(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return list(output.values())

    def forward(self, pred_im, gt):
        loss = []
        pred_im = (pred_im + 1) / 2
        gt = (gt + 1) / 2 # to [-1,1]
        normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pred_im = normalize(pred_im)
        gt = normalize(gt)
        pred_im_features = self.output_features(pred_im)
        gt_features = self.output_features(gt)
        layer_weights = {"relu1_2": 0.2, "relu2_2": 0.5, "relu3_3": 0.3}
        for name, (pred_im_feature, gt_feature) in zip(self.layer_name_mapping.values(), zip(pred_im_features, gt_features)):
            loss.append(layer_weights[name] * F.mse_loss(pred_im_feature, gt_feature))

        return sum(loss)/len(loss)

def compute_edges(image):
    if image.shape[1] > 1:
        edge_magnitude = torch.zeros_like(image)
        for c in range(image.shape[1]):
            edge_magnitude[:, c:c+1, :, :] = sobel(image[:, c:c+1, :, :])
    else:
        edge_magnitude = sobel(image)
    
    return edge_magnitude

def edgeloss(pred_image, gt_image):
    """
    Compute edge loss using Kornia Sobel operator for edge detection.

    Args:
        pred_image (torch.Tensor): Predicted image tensor (B, C, H, W).
        gt_image (torch.Tensor): Ground truth image tensor (B, C, H, W).

    Returns:
        torch.Tensor: Edge loss (scalar).
    """
    # 计算预测图像和 GT 图像的边缘强度
    edge_pred = compute_edges(pred_image)
    edge_gt = compute_edges(gt_image)

    # 计算边缘之间的 L1 损失
    return F.l1_loss(edge_pred, edge_gt)


