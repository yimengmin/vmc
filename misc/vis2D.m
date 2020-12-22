filename = 'Excited_State_3.txt';
psi=importdata(filename);
psi = reshape(psi,[500,500]);
psi = psi.^2;
x = -12:24/(500-1):12;
y = -12:24/(500-1):12;
[X,Y] = meshgrid(x,y);
% surf(X,Y,psi)
h= figure;
contourf(X,Y,psi)
% contourf(X,Y,psi,[0 0,02 0,04 0.06 0.08 0.1 0.12 0.14 0.16 0.18 0.2])
shading interp
colorbar
xlim([-3 3])
ylim([-3 3])
caxis([0 0.2])
% colormap
set(gca,'FontSize',20)
% colormap summer
% colormap( flipud(gray(256)) )
dir = "/Users/moe/Desktop/"
saveas(gcf,dir+'2DS3','epsc')
