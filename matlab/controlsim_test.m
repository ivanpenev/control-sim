n = int64(20);
m = int64(9);

A = rand_mat(n, -5, -1);
fprintf('Maximal real part of an eigenvalue of A: %g\n', max(real(eig(A))));

B = rand_orth(n, m);
fprintf('Norm of Bt*B - Im: %g\n', norm(B'*B - eye(m), 'fro'));

BBt = B*B';
gamma = 0.5;

G = rand_perturb(A, 0.75);
fprintf('Maximal singular value of inv(L_At)(Gt*G): %g\n', max(svd(lyap(A', G'*G))));

Q = rand_sym(n, 1, 4);
fprintf('Norm of Q - Qt: %g\n', norm(Q - Q', 'fro'));
fprintf('Minimal singular value of Q: %g\n', min(svd(Q)));

x0 = rand(n, 1);
L = x0 * x0';

kmax = int64(20);
T = zeros(n*(n+1)/2, n*(n+1)/2);

[P, nstep] = gen_care(A, G, gamma, BBt, Q, T, kmax);
fprintf('Norm of At*P + P*A + Gt*P*G - gamma*P*B*Bt*P + Q: %g\n',...
    norm(A'*P + P*A + G'*P*G - gamma*P*BBt*P + Q, 'fro'));
fprintf('Solution computed in %d iterations\n', nstep);

S = A - gamma*BBt*P;

R = gen_lyap(S, G, L, T);
fprintf('Norm of (A-gamma*B*Bt*P)*R + R*(A-gamma*B*Bt*P)t + G*R*Gt + L: %g\n',...
    norm(S*R + R*S' + G*R*G' + L, 'fro'));

M = P*R*P;
D = BBt*M - M*BBt;
GradB = gamma*(BBt*D - D*BBt);

%nGradB = -gamma*((M - BBt*M)*BBt + BBt*(M - M*BBt));
nGradB = mincost_ngrad(0, BBt(:), gamma, A, G, Q, L, T, kmax);
fprintf('Norm of nGradB + gamma*[B*Bt, [B*Bt, P*R*P]]: %g\n', norm(nGradB + GradB(:)));

fprintf('Norm of the gradient at B*Bt: %g\n', norm(nGradB));

tf = 1000;
sol = ode45(@(s, X) mincost_ngrad(s, X, gamma, A, G, Q, L, T, kmax), [0, tf], BBt);

fprintf('Norm of the gradient at the end of the time interval: %g\n',...
    norm(mincost_ngrad(tf, deval(sol, tf), gamma, A, G, Q, L, T, kmax)));
