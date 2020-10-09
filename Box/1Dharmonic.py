import torch
import numpy as np
import torch.nn as nn
Scale = 20.0
N, D_in, H1,H2,H3,H4, D_out = 45360,1,16,8,8,16,1
STEPS = 30000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
interv = 500 # plot the energy every 50 steps
K = 1 # para term,center around x==0.5
'''
assume m = hbar = k = 1
'''
para_pot = lambda p_x: 0.5*K*p_x**2 
qnn = lambda pos:(pos-Scale)*(Scale+pos)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--level', default=1, type=int, help='Energy Level')
opt = parser.parse_args()
class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H1,H2,H3,H4, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H1)
    self.linear2 = torch.nn.Linear(H1, H2)
    self.linear3 = torch.nn.Linear(H2, H3)
    self.linear4 = torch.nn.Linear(H3, H4)
    self.linear5 = torch.nn.Linear(H4, D_out)
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
    h4 = torch.nn.Tanh()(self.linear4(h3))
    y_pred = self.linear5(h4)
    return y_pred
# N is batch size in our case, N replace the sampling density on 1D condition; D_in is input dimension;
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
model = TwoLayerNet(D_in, H1,H2,H3,H4, D_out)
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
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
loss_his = []
for t in range(STEPS):
#  psi = qnn(x) #pbc
#  y_der0 = torch.mul(psi.reshape(-1),model(x).reshape(-1))
  y_der0 = model(x).reshape(-1)
  ### normalize
  ampli = torch.sum(torch.mul(y_der0,y_der0)).float() * (2*Scale)/N
  y_der0 = y_der0/torch.sqrt(ampli) #Normalize wavefunction
  y_der1= (y_der0[1:N]-y_der0[0:N-1])*N/(2*Scale) 
  y_der2 = (y_der1[1:N-1]-y_der1[0:N-2])*N/(2*Scale)
  y_der2.reshape(-1)
  # Compute and print loss
  # para_pot is the potential function
  parabola_pot = torch.sum(torch.mul(para_pot(x).reshape(-1),torch.mul(y_der0,y_der0))).float()
  loss = -0.5*torch.sum(torch.mul(y_der2,y_der0[0:N-2])).float() + parabola_pot # add  p^2
  loss = loss* 2*Scale/N # use sum to replace the inte

  #loss = loss_fn(y_pred, y)
  if (t%interv==0):
      print(t, loss.item(),loss.item())
      loss_his.append(loss.item())
  # Zero gradients, perform a backward pass, and update the weights.
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
nn_value = y_der0.cpu().detach().numpy()
loss_val = loss.cpu().detach().numpy()
from matplotlib import pyplot as plt
plt.figure()
plt.plot(sampling_value,nn_value, 'r', label='Energy : %.5f'%loss_val)
#gt_solution = np.sin(sampling_value*np.pi*e_level)
#gt_solution = gt_solution/np.sqrt(sum(gt_solution**2))
#plt.plot(sampling_value,gt_solution, 'g', label='Ground Truth Solution')
plt.legend(loc='best')
plt.savefig('Ground_State.png')
plt.clf()
plt.plot(np.arange(0,int(STEPS/interv),1),loss_his,label='Training Loss')
plt.legend(loc='best')
plt.xlabel('Steps per %d' %interv)
plt.ylabel('Energy')
np.savetxt('Ground_State.txt',nn_value)
