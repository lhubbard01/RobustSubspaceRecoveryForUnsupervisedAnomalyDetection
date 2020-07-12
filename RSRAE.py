import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




















def dim2(in_channels : int, out_channels : int, kernel_size, act : nn.Module):
  return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            act()
        )

def dim1(in_channels : int, out_channels : int, kernel_size, act : nn.Module):
  return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size),
            act()
        )

def dim2Up(in_channels : int, out_channels : int,kernel_size,act : nn.Module):
  return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size),
            act()
        )

def dim1Up(in_channels : int, out_channels : int, kernel_size : int, act : nn.Module):
  return nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size),
            act()
        )





def gen_enc_dec(dim : str, act : nn.Module, x : int, h : int, z : int, eOrD=None):
  if eOrD is None:
    if dim == "1":
      encode, decode = dim1, dim1Up

    elif dim == "2":
      encode, decode = dim2, dim2Up

    enc = nn.ModuleList(
                [ encode(x, h, 4, act),
                  encode(h, h, 4, act),
                  encode(h, z, 4, act)])

    dec = nn.ModuleList(
                [ decode(z, h, 4, act),
                  decode(h, h, 4, act),
                  decode(h, x, 4, act)])
    
    return enc, dec

  elif eOrD == "en":
    if dim == "1":   
        return nn.ModuleList(
            [ dim1(x, h, 4, act),
              dim1(h, h, 4, act),
              dim1(h, z, 4, act)])

    elif dim == "2":
        return nn.ModuleList(
            [ dim2(x, h, 4, act),
              dim2(h, h, 4, act),
              dim2(h, z, 4, act)])
    else:
        raise ValueError("Unexpected argument for dim, expects string of \"1\" or \"2\", got " + str(dim) )



class AutoEncoder(nn.Module):
  def __init__(self, en : nn.ModuleList, de : nn.ModuleList):
    super(AutoEncoder, self).__init__()
    self.encoder = en
    self.decoder = de

  def forward(self,X):
    for module in self.encoder:
      X = module(X)
    
    for module in self.decoder:
      X = module(X)
    return X

class L2Norm(nn.Module):

  def __init__(self):
    super(L2Norm,self).__init__()
  
  def forward(self,X):
    return torch.sqrt(torch.pow(X, 2).sum(0))
  










class RSRAE(AutoEncoder):
  """Robust Subspace Recovery Auto Encoder"""
  def __init__(self, 
               en  : nn.Module, 
               de  : nn.Module, 
               dim : int,
               rsrlayer: nn.Module, 
               lambda1 : float=0.1, 
               lambda2 : float=0.1):
      

    super(RSRAE, self).__init__(en, de)
    self.rsrlayer = rsrlayer
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.dim = dim

    self.MSE = nn.MSELoss()
    self.l2norm = L2Norm()
  


  def forward(self, X):
    for module in self.encoder:
      X = module(X)

    self.z = X #need to save for loss function (which I didnt write an autograd extension for)
    if self.dim == 2:
        self.z = self.z.reshape(-1, 4 * 19 * 19)
    
    self.z_til = self.rsrlayer(self.z) # linear transformation into further subspace, by the learned orthogonal matrix to separate the inliers from the outliers
    
    X_hat = self.z_til
    if self.dim == 2:
        X_hat = X_hat.reshape(-1, 4, 5, 5) # note the convolution FROM the further compressed space, instead of transforming back to z's dim

    for module in self.decoder:
      X_hat = module(X_hat)

    return X_hat
  
  def loss(self,out,d):
    rsr = [param for param in self.rsrlayer.parameters()][0] #since this is a generator, maybe theres a better way to access?

    AEloss = self.MSE(out, d) #AutoEncoder Loss (reconstruction error)

    inner_prod_of_z_transformed = torch.matmul(rsr.T, self.z_til.T) 
    l2norm_latent_inner_sum = self.l2norm(self.z - inner_prod_of_z_transformed.T).sum(keepdim=True,dim=0)
    outer_prod = torch.matmul(rsr, rsr.T)
    rsr_loss1 = self.lambda1 * l2norm_latent_inner_sum  # requisite steps to obtain loss for rsr1
    
    frob_norm = torch.norm(outer_prod - torch.eye(rsr.size(0)).cuda()) #frobenius norm of the outerp, less the identity
    rsr_loss2 = self.lambda2 * torch.pow(frob_norm, 2) #requisite steps to obtain loss for rsr2

    return AEloss, AEloss + rsr_loss1 + rsr_loss2, rsr_loss1, rsr_loss2 # AELoss, RSRAE+, RSR1, RSR2
       
