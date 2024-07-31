import torch
import torch.nn as nn
import torch.nn.functional as func
import cv2
import numpy as np

# Pytorch-implemented functions:
# MSE (L2)
# MAE (L1)
# Huber (tested with reduction = 'mean', delta = 0.75)
    
class RiemannianLoss(nn.Module):
    def __init__(self, gamma):
        super(RiemannianLoss, self).__init__()
        self.gamma = gamma
    
    def forward(self, input, target):
        # Ensure inputs are non-zero to avoid NaN in logarithms
        eps = torch.finfo(input.dtype).eps  # small epsilon to avoid division by zero
        input = torch.clamp(input, min=eps)
        target = torch.clamp(target, min=eps)
        
        # Calculate the absolute differences in logarithms
        abs_log_diff = torch.abs(torch.log(torch.abs(target)) - torch.log(torch.abs(input)))
        
        # Compute the loss function
        loss = torch.mean(torch.exp(self.gamma * abs_log_diff))
        
        return loss
    
# sigma (called alpha in the original paper) was set to 0.7 in the paper  
class EdgeLoss(nn.Module):
    def __init__(self, sigma, dilate=False, kernel_size=2, low_threshold=75, high_threshold=150):
        super(EdgeLoss, self).__init__()
        self.sigma = sigma
        self.dilate = dilate
        self.kernel_size = kernel_size
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def edgemap(self, image_tensor):

        if image_tensor.dim() > 3:
            image_tensor = image_tensor.squeeze(0)

        # Convert PyTorch tensor to NumPy array (assuming RGB image)
        image_np = image_tensor.permute(1, 2, 0).cpu().numpy()  # (C, H, W) -> (H, W, C)
        image_np = image_np - np.min(image_np)
        image_np = image_np / np.max(image_np)
        image_np = (image_np * 255).astype(np.uint8)  # Scale to 0-255 and convert to uint8

        # Convert RGB to BGR (OpenCV expects BGR format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Apply Canny edge detection
        edges = cv2.Canny(image_bgr, self.low_threshold, self.high_threshold)
        if self.dilate:
            kernel = np.ones((self.kernel_size,self.kernel_size), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)

        # Convert NumPy array back to PyTorch tensor
        edges_tensor = torch.tensor(edges / 255.0, dtype=torch.float32)  # Normalize to [0, 1]
        
        return edges_tensor
    
    def forward(self, input, target):
        H = target.size(2)
        W = target.size(3)
        
        edge_maps = []
        
        batch_size = target.size(0)
        for i in range(batch_size):
            
            edge_map = self.edgemap(target[i])            
            edge_maps.append(edge_map)
                   
        edge_maps = torch.stack(edge_maps, dim=0).cuda()

        num_edge_pixels = torch.sum(edge_maps).item()
        num_total_pixels = H * W
        diff_img = torch.abs(input - target)

        loss_edges = (torch.sum(edge_maps * diff_img))/num_edge_pixels
        loss_pixels = (torch.sum(diff_img))/num_total_pixels
        loss_total = (self.sigma)*loss_pixels + (1-self.sigma)*loss_edges
        
        return loss_total


class Charbonnier_Loss:
    """
    Charbonnier Loss (L1)
    """
    def __init__(self, complex_i=False, eps=1e-3):
        """
        @args:
            - complex_i (bool): whether images are 2 channelled for complex data
            - eps (float): epsilon, different values can be tried here
        """
        self.complex_i = complex_i
        self.eps = eps

    def __call__(self, outputs, targets, weights=None):

        B, C, T, H = targets.shape#, W = targets.shape
        if(self.complex_i):
            assert C==2, f"Complex type requires image to have C=2, given C={C}"
            diff_L1_real = torch.abs(outputs[:,0]-targets[:,0])
            diff_L1_imag = torch.abs(outputs[:,1]-targets[:,1])
            loss = torch.sqrt(diff_L1_real * diff_L1_real + diff_L1_imag * diff_L1_imag + self.eps * self.eps)
        else:
            diff_L1 = torch.abs(outputs-targets)
            loss = torch.sqrt(diff_L1 * diff_L1 + self.eps * self.eps)

        if(weights is not None):

            if(weights.ndim==1):
                weights = weights.reshape(B,1,1,1,1)
            elif weights.ndim==2:
                weights = weights.reshape(B,T,1,1,1)
            else:
                raise NotImplementedError(f"Only support 1D(Batch) or 2D(Batch+Time) weights for L1_Loss")

            v_l1 = torch.sum(weights*loss) / torch.sum(weights)
        else:
            v_l1 = torch.sum(loss)

        return v_l1 / loss.numel()
 

class GeneralizedLoss(nn.Module):
    def __init__(self, alpha, c):
        super(GeneralizedLoss, self).__init__()
        self.alpha = alpha
        self.c = c

    def forward(self, outputs, targets):
        x = torch.abs(outputs - targets)
        if self.alpha == 2:
            return torch.mean(0.5*((x/(self.c))**2))
        elif self.alpha == 0:
            return torch.mean(torch.log10(0.5*((x/(self.c))**2) + 1))
        elif self.alpha < 0 and np.isinf(self.alpha):
            return torch.mean(1-(np.exp(-0.5*((x/(self.c))**2))))
        else:    
            term1 = max(self.alpha - 2.0, 2.0 - self.alpha) / self.alpha  # Ensure alpha - 2.0 is a float
            term2 = ((x / self.c)**2 / abs(self.alpha - 2) + 1)**(self.alpha / 2) - 1
            loss = term1 * term2
            return loss.mean()


class PSNR(nn.Module):
    """
    PSNR as a comparison metric
    """
    def __init__(self, range=1.0):
        super(PSNR, self).__init__()

        """
        @args:
            - range (float): max range of the values in the images
        """
        self.range=range

    def __call__(self, outputs, targets):

        num = self.range * self.range
        den = torch.mean(torch.square(targets - outputs)) + torch.finfo(torch.float32).eps

        return -1 * 10 * torch.log10(num/den)
    
