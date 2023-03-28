function [W,H,iter,time,timeaxis,objstr] = pnm(V,Winit,Hinit,tol,timelimit,maxiter)
% Solve nonnegative matrix factorization problem:
% min_W,H 1/2*||V - WH||_F^2  s.t. W_ij >= 0, H_ij >= 0

% ------------------ INPUT ------------------------
% V: nonnegative constant matrix
% Winit,Hinit: initial solution
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

% ----------------- OUTPUT ------------------------
% W,H: output solution

% Author: Hanju Wu, Sun Yat-sen University
% Part of this function is modified based on Prof. Chih-Jen Lin's nmf.m and
% Dr. Pinghua Gong's pnm_nmf.m

W = Winit; H = Hinit; tic; etime = 0;

E = W*H - V;
objnow = .5*norm(E,'fro')^2;

gradW = W*(H*H') - V*H'; gradH = (W'*W)*H - W'*V;

tol = max(0.001,tol);%*initgrad; Yue changed this to absolute error
objstr = [];
timeaxis = [];
projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
sigma = 1e-3; beta = 5e-1; epsilon = 1e-12;
for iter=1:maxiter
    if toc > timelimit || projnorm < tol%Yue changed the stopping criterion
        break;
    end

    PkH = compute(V,W,H,epsilon);
    PkW = compute(V',H',W',epsilon);
    PkW = PkW';

    % search step size
    mk = 0;
    for inner_iter=1:20
        Hn = max(0,H - beta^mk*PkH);
        Wn = max(0,W - beta^mk*PkW);
        En = Wn*Hn - V; objnew = .5*norm(En,'fro')^2;
        if objnew - objnow < sigma*sum(sum([Wn' - W',Hn - H].*[gradW',gradH]))
            W = Wn; H = Hn; E = En; objnow = objnew;
            break;
        else
            mk = mk + 1;
        end 
    end
    gradW = E*H'; gradH = W'*E;
    temp = toc;
    objstr = [objstr;objnow];
    etime = etime + toc - temp;
    timeaxis = [timeaxis,toc - etime];
    projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
end
time = toc - etime;
fprintf('\nFinal Iteration %d\n', iter);
end

function Pk = compute(V,W,H,epsilon)
% Author: Hanju Wu, Sun Yat-sen University

[r,n] = size(H);
WtW=W'*W;
gradW = W*(H*H') - V*H';
gradH = (W'*W)*H - W'*V;
w=norm(W - max(0,W - gradW),'fro');
h=norm(H - max(H - gradH,0),'fro');
wk=sqrt(w^2+h^2);
epsilonk = min(epsilon,wk);
INDk = ((H >= 0) & (H <= epsilonk) & (gradH > 0));
invINDk = xor(ones(r,n),INDk);
Pk = computePk(WtW,gradH,INDk,invINDk,r,n);
end

function Pk = computePk(WtW,GRAD,INDk,invINDk,r,m)
% Compute Pk: a subfunction of pnm_nlssubprob
% Author: Hanju Wu, Sun Yat-sen University
Pk = zeros(r,m);
if all(invINDk(:))
    Pk = WtW\GRAD;
else
    tempWtW=WtW;
    for i=1:m
        for j=1:r
            if INDk(j,i)
                temp=WtW(j,j);
                WtW(:,j)=zeros(1,r)';
                WtW(j,:)=zeros(1,r);
                WtW(j,j)=temp;
            end
        end
        Pk(:,i)=WtW\GRAD(:,i);
        WtW=tempWtW;
    end
end
end