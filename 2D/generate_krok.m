g = 600;
e = ones(g,1);
L = spdiags([e -2*e e], -1:1, g, g);
I = speye(g);
L3 = kron(L,I) + kron(I, L)
H =  -0.5 * L3;
save('Energy_operator.mat','H');
