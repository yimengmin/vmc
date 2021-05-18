import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
from scipy import sparse
import scipy.io as sio
import numpy as np 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
interv = 500 # plot the ebery every 50 steps

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--level', default=2, type=int, help='Energy Level')
parser.add_argument('--decay', default=1000, type=float, help='Orth Pene')
parser.add_argument('--depth', default=6, type=int, help='Depth')
parser.add_argument('--width', default=16, type=int, help='width')
parser.add_argument('--step_size', default=10000, type=int, help='step size for lr decay')
parser.add_argument('--lr', default=5e-3, type=int, help='lr')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--STEPS', default=100000, type=int, help='number of training steps')
parser.add_argument('--alpha', default=2, type=float, help='L_{\alpha} norm')
parser.add_argument('--scale', default=5, type=float, help='scale')
parser.add_argument('--DIM', default=500, type=int, help='dim')
opt = parser.parse_args()
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
np.random.seed(opt.seed)
torch.backends.cudnn.deterministic = True
STEPS=opt.STEPS


Scale = opt.scale
DIM = opt.DIM # the grid
unit_area = (2*Scale)**2/((DIM-1)**2)
minenergy = 1e6
def spy_sparse2torch_sparse(data):
    """
    :param data: a scipy sparse csr matrix
    :return: a sparse torch tensor
    """
    samples=data.shape[0]
    features=data.shape[1]
    values=data.data
    coo_data=data.tocoo()
    indices=torch.LongTensor([coo_data.row,coo_data.col])
    t=torch.sparse.FloatTensor(indices,torch.from_numpy(values).float(),[samples,features])
    return t
diag = np.ones([DIM])
diags = np.array([-diag, 2.0*diag, -diag])
kinetic_1d = sparse.spdiags(diags, np.array([1.0, 0.0, -1.0]), DIM, DIM)
T = sparse.kronsum(kinetic_1d, kinetic_1d)
H_matrix = spy_sparse2torch_sparse(T)
H_matrix = 0.5*H_matrix


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
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,weight_decay=4e-5)
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
  orth_pen = (torch.abs(torch.matmul(gs0,y_der0))*unit_area)**opt.alpha
  #  * (2*Scale)/N  normalize them here
  for indx in range(1,opt.level):
      gs_previous = globals()['excited_state_'+str(indx)]
      orth_pen = orth_pen+(torch.abs(torch.matmul(gs_previous,y_der0))*unit_area)**opt.alpha

  energy = torch.sum(torch.mul(laplacian,y_der0)).float() *unit_area # int->sum
  energy = energy + potential_energy 
  loss = opt.decay*(orth_pen) + energy
  if (energy.item()<minenergy)and(t>2000):
      print(t,minenergy)
      minenergy = energy.item()
      nn_value = y_der0.cpu().detach().numpy()
      np.savetxt('Excited_State_%sw%dd%d.txt'%(opt.level,opt.width,opt.depth),nn_value)
  if (t%interv==0):
      print(t,energy.item(),loss.item())
      loss_his.append(loss.item())
  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  scheduler.step()
print('Energy value: %.10f'%minenergy)

#nn_value = y_der0.cpu().detach().numpy()
#psi_matrix = np.zeros((DIM,DIM))
#for i in range(DIM):
#  for j in range(DIM):
#      psi_matrix[i][j] = nn_value[i*DIM+j]

#import matplotlib
#matplotlib.use('Agg') # for saving the figure
#from matplotlib import pyplot as plt
#plt.figure()
#x = np.outer(np.linspace(-1*Scale, Scale, DIM), np.ones(DIM))
#y = x.copy().T # transpose
#plt = plt.axes(projection='3d')
#plt.plot_surface(x,y, psi_matrix,cmap='viridis', edgecolor='none')
#plt.set_title('Energy:%.5f'%energy.cpu().detach().numpy())
#plt.figure.savefig('Excited_State_%sw%dd%d.png'%(opt.level,opt.width,opt.depth)) # for 2D
#np.savetxt('Excited_State_%sw%dd%d.txt'%(opt.level,opt.width,opt.depth),nn_value)
