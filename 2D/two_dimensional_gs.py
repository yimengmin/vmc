import torch
Scale = 12.0
DIM = 500 # the grid
unit_area = (2*Scale)**2/((DIM-1)**2)
import torch.nn as nn
STEPS = 20000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
interv = 5 # plot the ebery every 50 steps
import numpy as np 
import scipy.io as sio
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


class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H1,H2,H3,H4,H5,H6,H7,H8, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H1)
    self.linear2 = torch.nn.Linear(H1, H2)
    self.linear3 = torch.nn.Linear(H2, H3)
    self.linear4 = torch.nn.Linear(H3, H4)
#    self.linear5 = torch.nn.Linear(H4, H5)
#    self.linear6 = torch.nn.Linear(H5, H6)
#    self.linear7 = torch.nn.Linear(H6, H7)
#    self.linear8 = torch.nn.Linear(H7, H8)
    self.linear9 = torch.nn.Linear(H4, D_out)
  def forward(self, x):
    """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary (differentiable) operations on Tensors.
    """
    #h_relu = self.linear1(x).clamp(min=0)
    h1 = torch.nn.Tanh()(self.linear1(x))
    h2 = torch.nn.Tanh()(self.linear2(h1))
    h3 = torch.nn.Tanh()(self.linear3(h2))
#    h4 = torch.nn.Tanh()(self.linear4(h3))
#    h5 = torch.nn.Tanh()(self.linear5(h4))
#    h6 = torch.nn.Tanh()(self.linear5(h5))
#    h7 = torch.nn.Tanh()(self.linear5(h6))
    h8 = torch.nn.Tanh()(self.linear4(h3))
    y_pred = self.linear9(h8)
    return y_pred

N, D_in, H1,H2,H3,H4,H5,H6,H7,H8, D_out =DIM**2,2,16,16,16,16,16,16,16,16,1
inv = 2*Scale/(DIM-1)
pos_matrix = np.zeros((DIM**2,2))  # the 2D position array 
for i in range(DIM):
  for j in range(DIM):
      pos_matrix[i*DIM+j] = (i*inv-6.,j*inv-6.)
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
model = TwoLayerNet(D_in, H1,H2,H3,H4,H5,H6,H7,H8, D_out)
model=model.to(device)
H_matrix = H_matrix.float().to(device)
## ini
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-2)
loss_his = []
for t in range(STEPS):
  y_der0 = model(input_pos)
  ampli = torch.sum(torch.mul(y_der0,y_der0)).float()*unit_area  # int->sum
  y_der0 = y_der0/torch.sqrt(ampli)
  laplacian = torch.spmm(H_matrix,y_der0)/unit_area  # (DIM^2 \times 1)
  
  potential_energy = torch.sum(torch.mul(y_der0,torch.mul(y_der0,potential_matrix))).float()*unit_area

  loss = torch.sum(torch.mul(laplacian,y_der0)).float() *unit_area # int->sum
  loss = loss + potential_energy 
  if (t%interv==0):
      print(t,loss.item())
      loss_his.append(loss.item())
  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

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
plt.set_title('Energy:%.5f'%loss.cpu().detach().numpy())
plt.figure.savefig('Ground_State.png')
np.savetxt('Ground_State.txt',nn_value)
