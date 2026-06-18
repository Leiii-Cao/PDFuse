import torch
import torch.fft
from torch.nn.functional import pad
from skimage import io, exposure, img_as_ubyte, img_as_float
from tqdm import trange
import numpy as np

class LIME_PyTorch:
    def __init__(self, iterations=1, alpha=2, rho=2, gamma=0.7, strategy=2):
       
        self.iterations = iterations
        self.alpha = alpha
        self.rho = rho
        self.gamma = gamma
        self.strategy = strategy

    def first_order_derivative(self, n, k=1, device='cpu'):
        """
        Construct first-order derivative matrix with diagonal shifting
        
        Args:
            n (int): Matrix dimension (n x n)
            k (int): Diagonal offset (positive for upper, negative for lower)
            device (str): Target computation device ('cpu' or 'cuda')
            
        Returns:
            torch.Tensor: Derivative matrix of shape (n, n)
        """
        main_diag = -torch.eye(n, device=device)
        shifted_diag = torch.diag(torch.ones(n-1, device=device), diagonal=k)
        return main_diag + shifted_diag

    def toeplitz_matrix(self, n, row, device='cpu'):
        """
        Construct Toeplitz matrix for frequency-domain optimization
        
        Args:
            n (int): Total number of elements in flattened matrix
            row (int): Number of rows in original 2D shape
            device (str): Target computation device
            
        Returns:
            torch.Tensor: Toeplitz matrix in 2D shape (row, n//row)
        """
        vecDD = torch.zeros(n, device=device)
        vecDD[0] = 4
        vecDD[1] = -1
        vecDD[row] = -1
        vecDD[-1] = -1
        vecDD[-row] = -1
        return vecDD

    def vectorize(self, tensor):
        # For a 2D tensor, tensor.t() returns its transpose.
        # Then .reshape(-1) flattens it into a one-dimensional tensor.
        return tensor.t().reshape(-1)
        
    def load(self, tensor, device='cpu'):
        """
        Load image and initialize computational components
        
        Args:
            img_path (str): Path to input image file
            device (str): Target computation device
        """
        # Read and convert image to HWC tensor
        self.L = tensor # (H, W, 3)
        
        # Extract spatial dimensions
        self.h, self.w, _ = self.L.shape
        
        # Initial illumination estimate (max across channels)
        self.T_hat, _ = torch.max(self.L, dim=2)  # (H, W)
        
        # Construct derivative operators
        self.dv = self.first_order_derivative(self.h, device=device)  # (H, H)
        self.dh = self.first_order_derivative(self.w, k=-1, device=device)  # (W, W)
        # Build Toeplitz system matrix
        self.vec_dd = self.toeplitz_matrix(self.h*self.w, self.h, device=device)  # (H, W)
        
        # Initialize weighting matrix
        self.W = self.weighting_strategy(device)
    def weighting_strategy(self, device):
        """
        Compute adaptive weighting matrix based on selected strategy
        
        Args:
            device (str): Target computation device
            
        Returns:
            torch.Tensor: Weight matrix of shape (2H, W)
        """
        if self.strategy == 2:
            # Compute directional gradients
            dTv = torch.einsum('ik,kj->ij', self.dv, self.T_hat)  # (H, H) @ (H, W) -> (H, W)
            dTh = torch.einsum('ik,kj->ij', self.T_hat, self.dh)  # (H, W) @ (W, W) -> (H, W)
            # Calculate adaptive weights
            Wv = 1 / (torch.abs(dTv) + 1)  # (H, W)
            Wh = 1 / (torch.abs(dTh) + 1)  # (H, W)
            
            return torch.cat([Wv, Wh], dim=0)  # (2H, W)
        else:
            return torch.ones((2*self.h, self.w), device=device)

    def _T_subproblem(self, G, Z, u):
        """
        Solve illumination subproblem using frequency-domain method
        
        Args:
            G (torch.Tensor): Auxiliary variable
            Z (torch.Tensor): Lagrange multipliers
            u (float): Penalty parameter
            
        Returns:
            torch.Tensor: Updated illumination map (H, W)
        """
        # Split vertical/horizontal components
        X = G - Z / u
        Xv = X[:self.h, :]  # (H, W)
        Xh = X[self.h:, :]  # (H, W)
        
        # Compute spatial gradient term
        spatial_term = torch.einsum('ik,kj->ij', self.dv, Xv) + torch.einsum('ik,kj->ij', Xh, self.dh)
        # Frequency-domain computation
        numerator = torch.fft.fft(self.vectorize(2*self.T_hat + u*spatial_term))
        denominator = torch.fft.fft(self.vec_dd* u)  + 2
        
        T = torch.fft.ifft(numerator / denominator)
        T = T.view(self.w, self.h).real.T
        # Numerical stabilization
        return self.rescale_intensity(T, (0, 1), (0.001, 1))

    def _G_subproblem(self, T, Z, u, W):
        """
        Solve auxiliary variable subproblem with shrinkage operator
        
        Args:
            T (torch.Tensor): Current illumination estimate
            Z (torch.Tensor): Lagrange multipliers
            u (float): Penalty parameter
            W (torch.Tensor): Weight matrix
            
        Returns:
            torch.Tensor: Updated auxiliary variable (2H, W)
        """
        # Compute gradients

        dTv = torch.einsum('ik,kj->ij', self.dv, T)  # (H, H) @ (H, W) -> (H, W)
        dTh = torch.einsum('ik,kj->ij', T, self.dh)  # (H, W) @ (W, W) -> (H, W)
        dT = torch.cat([dTv, dTh], dim=0)  # (2H, W)
        
        # Shrinkage operation
        epsilon = self.alpha * W / u
        X = dT + Z / u
        return torch.sign(X) * torch.clamp(torch.abs(X) - epsilon, min=0)



    def rescale_intensity(self,tensor, in_range=(0, 1), out_range=(0.001, 1)):
   
        tensor = tensor.float()
        in_min, in_max = in_range
        out_min, out_max = out_range
        scale = (out_max - out_min) / (in_max - in_min)
        tensor = (tensor - in_min) * scale + out_min
        tensor = torch.clamp(tensor, min=out_min, max=out_max)
        return tensor


    def enhance(self):
        """
        Main enhancement pipeline with ADMM optimization
        
        Returns:
            numpy.ndarray: Enhanced image in uint8 format (H, W, C)
        """
        # Initialize variables
        T = torch.zeros_like(self.T_hat)
        G = torch.zeros((2*self.h, self.w), device=self.L.device)
        Z = torch.zeros_like(G)
        u = torch.tensor(1.0, device=self.L.device)
        # ADMM iterations
        for _ in range(self.iterations):
            T = self._T_subproblem(G, Z, u)
            G = self._G_subproblem(T, Z, u, self.W)
            # Update dual variables
            Z += u * (torch.cat([
                torch.einsum('ik,kj->ij', self.dv, T),
                torch.einsum('ik,kj->ij', T, self.dh)
            ], dim=0) - G)
            u *= self.rho
        # Reflectance calculation
        T_gamma = T** self.gamma
        R = self.L / T_gamma.unsqueeze(2)  # (H, W, 3)
        
        # Post-processing
        R = torch.clamp(R, 0, 1)

        return R.permute(2,0,1).unsqueeze(0),T_gamma.unsqueeze(2).permute(2,0,1).unsqueeze(0)



