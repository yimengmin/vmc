filename = 'Excited_State_9.txt';
psi=importdata(filename);
psi = reshape(psi,[600,600]);
psi = psi.^2;
x = -6:12/(600-1):6;
y = -6:12/(600-1):6;
[X,Y] = meshgrid(x,y);
% surf(X,Y,psi)
h= figure;
contourf(X,Y,psi)
shading interp
colorbar
xlim([-3 3])
ylim([-3 3])

% colormap
set(gca,'FontSize',20)
% colormap summer
% colormap( flipud(gray(256)) )
dir = "/Users/moe/comp_phys/neural_quantum_state/EXCITED NERONS/PRE/2DData/"
saveas(gcf,dir+'2DS9','epsc')
