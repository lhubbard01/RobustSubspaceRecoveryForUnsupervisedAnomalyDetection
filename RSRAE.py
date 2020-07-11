import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def dim2(indim,outdim,ks,act):
  return nn.Sequential(nn.Conv2d(indim,outdim,ks),act())
def dim1(indim,outdim,ks,act):
  return nn.Sequential(nn.Conv1d(indim,outdim,ks),act())

def dim2Up(indim,outdim,ks,act):
  return nn.Sequential(nn.ConvTranspose2d(indim,outdim,ks),act())
def dim1Up(indim,outdim,ks,act):
  return nn.Sequential(nn.ConvTranspose1d(indim,outdim,ks),act())





def genEncDec(dim:str, act:nn.Module, x:int, h:int, z:int, eOrD=None):
  if eOrD is None:
    if dim == "1":
      return nn.ModuleList([dim1(x,h,4,act),dim1(h,h,4,act),dim1(h,z,4,act)]), \
           nn.ModuleList([dim1Up(z,h,4,act), dim1Up(h,h,4,act), dim1Up(h,x,4,act)])
    if dim == "2":
      return nn.ModuleList([dim2(x,h,4,act),dim2(h,h,4,act),dim2(h,z,4,act)]), \
           nn.ModuleList([dim2Up(z,h,4,act), dim2Up(h,h,4,act), dim2Up(h,x,4,act)])

  elif eOrD == "en":
    if dim == "1":   return nn.ModuleList([dim1(x,h,4,act),dim1(h,h,4,act),dim1(h,z,4,act)])
    elif dim == "2": return nn.ModuleList([dim2(x,h,4,act),dim2(h,h,4,act),dim2(h,z,4,act)])

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
  def __init__(self, 
               en : nn.Module, 
               de : nn.Module, 
               dim:int,
               rsrlayer: nn.Module, 
               lambda1:float=0.5, 
               lambda2:float=0.5):
      

    super(RSRAE,self).__init__(en,de)
    self.rsrlayer = rsrlayer
    self.lambda1 = lambda1
    self.lambda2 = lambda2
    self.MSE = nn.MSELoss()
    self.dim = dim
    self.l2norm = L2Norm()
  


  def forward(self,X):
    for module in self.encoder:
      X = module(X)

    self.z = X
    bn = self.z.shape[0]
    if self.dim == 2: self.z = self.z.reshape(bn,4*19*19)
    self.z_til = self.rsrlayer(self.z)
    X_hat = self.z_til
    if self.dim == 2: X_hat = X_hat.reshape(bn,4,5,5)

    for module in self.decoder:
      X_hat = module(X_hat)
    return X_hat
  
  def loss(self,out,d):
    rsr = [param for param in self.rsrlayer.parameters()][0]
    AEloss = self.MSE(out, d)
    inner_prod_of_z_transformed = torch.matmul(rsr.T, self.z_til.T) 
    l2norm_latent_inner_sum = self.l2norm(self.z - inner_prod_of_z_transformed.T).sum(keepdim=True,dim=0)
    outer_prod = torch.matmul(rsr, rsr.T)
    frob_norm = torch.norm(outer_prod - torch.eye(rsr.size(0)).cuda()) #frobenius norm of the outerp, less the identity
    rsr_loss1 = self.lambda1 * l2norm_latent_inner_sum 
    rsr_loss2 = self.lambda2 * torch.pow(frob_norm, 2)
    return AEloss, AEloss + rsr_loss1 + rsr_loss2, rsr_loss1, rsr_loss2
       