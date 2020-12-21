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
H =  H * 100 + 0.5 * PE +  kron(diag(4 * V),I) + kron(I,diag(2 * V));

% return 10 smallest eigen
d = eigs(H,10,'smallestreal') % do not use 'smallestabs' here
[PSI,D] = eigs(H,10,'SA');  % do not use 'SM' here

AMP = reshape(PSI(:,1).^2,[100,100]);
% 1,2,3,4,5... corresponds to ground state, 1st excited....
[X,Y] = meshgrid(V,V);
h= figure;
contourf(X,Y,AMP)