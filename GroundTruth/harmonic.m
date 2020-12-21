% by Yimeng Min @ Cornell CS
% yimeng.min@gmail.com
g = 100;
e = ones(g,1);
L = spdiags([e -2*e e], -1:1, g, g);
I = speye(g);
L3 = kron(L,I) + kron(I, L);
H =  -0.5 * L3;

V = -5.0:0.1:4.9;
length(V.^2);
length(I);
P = diag(V.^2);
PE =  kron(P,I) + kron(I,P);
H =  H *100 + 0.5 * PE ;

% return 10 smallest eigrn
d = eigs(H,10,'smallestabs')
[PSI,D] = eigs(H,10,'SM');

AMP = reshape(PSI(:,5).^2,[100,100]);
% 1,2,3,4,5... corresponds to ground state, 1st excited....
[X,Y] = meshgrid(V,V);
h= figure;
contourf(X,Y,AMP)
