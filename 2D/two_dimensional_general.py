import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
Scale = 5.0
DIM = 500 # the grid
unit_area = (2*Scale)**2/((DIM-1)**2)
import torch.nn as nn

import scipy.io as sio
import numpy as np 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--level', default=2, type=int, help='Energy Level')
parser.add_argument('--decay', default=10, type=float, help='Orth Pene')
parser.add_argument('--depth', default=5, type=int, help='Depth')
parser.add_argument('--width', default=16, type=int, help='width')
parser.add_argument('--step_size', default=5000, type=int, help='step size for lr decay')
parser.add_argument('--lr', default=5e-3, type=int, help='lr')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--STEPS', default=100000, type=int, help='number of training steps')
parser.add_argument('--alpha', default=2, type=float, help='L_{\alpha} norm')
opt = parser.parse_args()
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
np.random.seed(opt.seed)
torch.backends.cudnn.deterministic = True
STEPS=opt.STEPS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
interv = 500 # plot the ebery every 50 steps
mat_contents = sio.loadmat('Energy_operator.mat')
print(mat_contents)
H_matrix = mat_contents['H']
print(H_matrix.shape)
import numpy as np
from scipy import sparse
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
H_matrix = sparse_mx_to_torch_sparse_tensor(H_matrix)



#load ground state
gs0 = np.loadtxt('Ground_Statew%dd%d.txt'%(opt.width,opt.depth))
gs0 = torch.from_numpy(gs0).float().to(device)
for indx in range(1,opt.level):
    previous_states = 'excited_state_'+str(indx) 
    globals()[previous_states] = torch.from_numpy(np.loadtxt('Excited_State_%dw%dd%d.txt'%(indx,opt.width,opt.depth))).float().to(device)


class MulLayerNet(torch.nn.Module):
  def __init__(self, D_in, num_layers, layer_size, D_out):
    super(MulLayerNet, self).__init__()
    self.linearstart = torch.nn.Linear(D_in, layer_size)
    self.linears = nn.ModuleList([nn.Linear(layer_size,layer_size) for i in range(num_layers)])
    self.linearend = torch.nn.Linear(layer_size, D_out)
#    self.activation = nn.LeakyReLU(0.2)
    self.activation = nn.Tanh()
  def forward(self, x):
    x = self.activation(self.linearstart(x))
    for i, l in enumerate(self.linears):
      x = self.activation(l(x))
    y_pred = self.linearend(x)
    return y_pred
N, D_in, D_out =DIM**2,2,1

#N, D_in, H1,H2,H3,H4,H5,H6,H7,H8, D_out =DIM**2,2,16,16,16,16,16,16,16,16,1
inv = 2*Scale/(DIM-1)
pos_matrix = np.zeros((DIM**2,2))  # the 2D position array 
for i in range(DIM):
  for j in range(DIM):
      pos_matrix[i*DIM+j] = (i*inv-Scale,j*inv-Scale)
input_pos = pos_matrix

# construct the square term correponds to the potential energy
potential_matrix = np.zeros((DIM**2,1))
for i in range(DIM):
  for j in range(DIM):
      potential_matrix[i*DIM+j] =0.5*((i*inv-Scale)**2+(j*inv-Scale)**2)
potential_matrix = torch.from_numpy(potential_matrix).float().to(device)

input_pos = torch.from_numpy(input_pos).float().to(device)
print('input_pos')
print(input_pos.size())
model = MulLayerNet(D_in,opt.depth,opt.width, D_out)
model=model.to(device)
H_matrix = H_matrix.float().to(device)
## ini
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=0.9)
loss_his = []
for t in range(STEPS+1):
  y_der0 = model(input_pos)
  ampli = torch.sum(torch.mul(y_der0,y_der0)).float()*unit_area  # int->sum
  y_der0 = y_der0/torch.sqrt(ampli)
  laplacian = torch.spmm(H_matrix,y_der0)/unit_area  # (DIM^2 \times 1)
    
  potential_energy = torch.sum(torch.mul(y_der0,torch.mul(y_der0,potential_matrix))).float()*unit_area

  #===================compute penality term of based on previous states============
  # 5 is just a constant to penlize gs more
  orth_pen = (torch.abs(torch.matmul(torch.t(gs0),y_der0))*unit_area)**opt.alpha
  #  * (2*Scale)/N  normalize them here
  for indx in range(1,opt.level):
      gs_previous = globals()['excited_state_'+str(indx)]
      orth_pen = orth_pen+(torch.abs(torch.matmul(torch.t(gs_previous),y_der0))*unit_area)**opt.alpha
      if t%1000==0:
        psi = y_der0.detach().cpu().numpy()
        np.savetxt('Excited_State_%sw%dd%d.txt'%(opt.level,opt.width,opt.depth),psi)
        print('============>Pfor GS:')
        print((torch.abs(torch.matmul(torch.t(gs0),y_der0))*unit_area)**opt.alpha)
        print('============>Penalty for state %d:'%indx)
        print(torch.abs(torch.matmul(torch.t(gs_previous),y_der0))*unit_area)
  #========================================

  energy = torch.sum(torch.mul(laplacian,y_der0)).float() *unit_area # int->sum
  energy = energy + potential_energy 
  loss = opt.decay*(orth_pen**(1./opt.alpha)) + energy
  if (t%interv==0):
      print(t,energy.item(),loss.item())
      loss_his.append(loss.item())
  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  scheduler.step()

nn_value = y_der0.cpu().detach().numpy()
psi_matrix = np.zeros((DIM,DIM))
for i in range(DIM):
  for j in range(DIM):
      psi_matrix[i][j] = nn_value[i*DIM+j]

import matplotlib
matplotlib.use('Agg') # for saving the figure
from matplotlib import pyplot as plt
plt.figure()
x = np.outer(np.linspace(-1*Scale, Scale, DIM), np.ones(DIM))
y = x.copy().T # transpose
plt = plt.axes(projection='3d')
plt.plot_surface(x,y, psi_matrix,cmap='viridis', edgecolor='none')
plt.set_title('Energy:%.5f'%energy.cpu().detach().numpy())
plt.figure.savefig('Excited_State_%sw%dd%d.png'%(opt.level,opt.width,opt.depth)) # for 2D
np.savetxt('Excited_State_%sw%dd%d.txt'%(opt.level,opt.width,opt.depth),nn_value)
