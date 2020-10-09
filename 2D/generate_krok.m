g = 500; g3 = g^3;
p = linspace(-6,6, g);         
[X, Y, Z] = meshgrid(p, p, p); 
h = p(2) - p(1);               
X = X(:); Y = Y(:); Z = Z(:);   
R = sqrt(X.^2 + Y.^2 + Z.^2);   
Vext = -1 ./ R;                
e = ones(g,1);               
% L = spdiags([e -2*e e], -1:1, g, g) / h^2; 
L = spdiags([e -2*e e], -1:1, g, g);

I = speye(g);
L3 = kron(L,I) + kron(I, L) 
% H = -0.5 * L3 + spdiags(Vext, 0, g3, g3);  
H =  -0.5 * L3;
save('Energy_operator.mat','H');