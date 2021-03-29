import torch
print(torch.cuda.is_available())
import numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
Scale = 20.0
N,D_in,D_out = 50000,1,1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
interv = 2000 # plot the energy every 50 steps
K = 1 # para term,center around x==0.5
'''
assume m = hbar = k = 1
'''
para_pot = lambda p_x: 0.5*K*p_x**2 
qnn = lambda pos:(pos-Scale)*(Scale+pos)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--level', default=1, type=int, help='Energy Level')
parser.add_argument('--depth', default=5, type=int, help='Depth')
parser.add_argument('--width', default=16, type=int, help='width')
parser.add_argument('--step_size', default=10000, type=int, help='step size for lr decay')
parser.add_argument('--lr', default=5e-3, type=int, help='lr')
parser.add_argument('--decay', default=2, type=float, help='Orth Pene')
parser.add_argument('--seed', default=42, type=int, help='seed')
parser.add_argument('--STEPS', default=100000, type=int, help='number of training steps')
opt = parser.parse_args()
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
np.random.seed(opt.seed)
torch.backends.cudnn.deterministic = True
STEPS=opt.STEPS
#load ground state
gs0 = np.loadtxt('Ground_Stated%dw%d.txt'%(opt.depth,opt.width))
gs0 = torch.from_numpy(gs0).float().to(device)
class MulLayerNet(torch.nn.Module):
  def __init__(self, D_in, num_layers, layer_size, D_out):
    super(MulLayerNet, self).__init__()
    self.linearstart = torch.nn.Linear(D_in, layer_size)
    self.linears = nn.ModuleList([nn.Linear(layer_size,layer_size) for i in range(num_layers)])
#    self.bns = nn.ModuleList([nn.BatchNorm1d(opt.width) for i in range(num_layers)])
    self.linearend = torch.nn.Linear(layer_size, D_out)
#    self.leakyrelu = nn.LeakyReLU(0.1)
    self.activation = nn.Tanh()
  def forward(self, x):
    x = self.activation(self.linearstart(x))
    for i, l in enumerate(self.linears):
      x = self.activation(l(x))
#      x = self.bns[i](x)
    y_pred = self.linearend(x)
    return y_pred
# H is hidden dimension; D_out is output dimension.
# Create random Tensors to hold inputs and outputs
sampling_value = np.linspace(-1*Scale,Scale,N)
#x = torch.randn(N, D_in) # N by 1 
input_x = sampling_value.reshape(-1,1)
x = torch.from_numpy(input_x).float().to(device) # change double to float
print(x.shape) # x represents the x coordinate
#y = torch.randn(N, D_out) # N by 1
#print(y.shape)
# Construct our model by instantiating the class defined above.
model = MulLayerNet(D_in,opt.depth,opt.width, D_out)

model=model.to(device)
## ini
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

model.apply(init_weights)
#y_der0 = model(x)
#y_der1= y_der0[1:N,:]-y_der0[0:N-1,:] 
#y_der2 = y_der1[1:N-1,:]-y_der0[0:N-2,:]

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
#loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
scheduler = StepLR(optimizer, step_size=opt.step_size, gamma=0.9)
loss_his = []
for t in range(STEPS+1):
#  psi = qnn(x) #pbc
#  y_der0 = torch.mul(psi.reshape(-1),model(x).reshape(-1))
  y_der0 = model(x).reshape(-1)
  ### normalize
  ampli = torch.sum(torch.mul(y_der0,y_der0)).float() * (2*Scale)/N
  y_der0 = y_der0/torch.sqrt(ampli) #Normalize wavefunction
  y_der1= (y_der0[1:N]-y_der0[0:N-1])*N/(2*Scale) 
  y_der2 = (y_der1[1:N-1]-y_der1[0:N-2])*N/(2*Scale)
  y_der2.reshape(-1)
  #===================compute penality term of based on previous states============
  orth_pen = (torch.abs(torch.sum(torch.mul(gs0,y_der0)))*(2*Scale)/N)**2
  #  * (2*Scale)/N  normalize them here
  #========================================

  # Compute and print loss
  # para_pot is the potential function
  parabola_pot = torch.sum(torch.mul(para_pot(x).reshape(-1),torch.mul(y_der0,y_der0))).float()
  energy = -0.5*torch.sum(torch.mul(y_der2,y_der0[0:N-2])).float() + parabola_pot # add  p^2
  energy = energy * 2*Scale/N # use sum to replace the inte
  loss = energy + opt.decay*orth_pen # add the penality term
  #loss = loss_fn(y_pred, y)
  if (t%interv==0):
      print(t, loss.item(),loss.item())
      loss_his.append(loss.item())
  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  scheduler.step()
nn_value = y_der0.cpu().detach().numpy()
loss_val = loss.cpu().detach().numpy()
from matplotlib import pyplot as plt
plt.figure()
plt.plot(sampling_value,nn_value, 'r', label='Energy : %.5f'%energy)
plt.legend(loc='best')
plt.savefig('Excited_State_1d%dw%d.png'%(opt.depth,opt.width))
plt.clf()
#plt.plot(np.arange(0,int(STEPS/interv)+1,1),loss_his,label='Training Loss')
#plt.legend(loc='best')
#plt.xlabel('Steps per %d' %interv)
#plt.ylabel('Energy')
np.savetxt('Excited_State_1d%dw%d.txt'%(opt.depth,opt.width),nn_value)
