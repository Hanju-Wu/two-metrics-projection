function [W,H,iter,time,timeaxis,objstr] = alspnm(V,Winit,Hinit,tol,timelimit,maxiter)
% Solve nonnegative matrix factorization problem:
% min_W,H 1/2*||V - WH||_F^2  s.t. W_ij >= 0, H_ij >= 0

% ------------------ INPUT ------------------------
% V: nonnegative constant matrix
% Winit,Hinit: initial solution
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

% ----------------- OUTPUT ------------------------
% W,H: output solution

% Written by Pinghua Gong, Tsinghua University
% Part of this function is modified based on Prof. Chih-Jen Lin's nmf.m

W = Winit; H = Hinit; tic; etime = 0;

gradW = W*(H*H') - V*H'; gradH = (W'*W)*H - W'*V;
temp = toc;
initgrad = norm([gradW; gradH'],'fro');
fprintf('Init gradient norm %f\n', initgrad);
etime = etime + toc - temp;
tolW = max(0.001,tol);%*initgrad; Yue changed this to absolute error
tolH = tolW;
objstr = [];
timeaxis = [];
projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);

for iter=1:maxiter
    if toc > timelimit || projnorm < tol%Yue changed the stopping criterion
        break;
    end
    
    [W,~,iterW] = pnm_nlssubprob(V',H',W',tolW,1000);
    W = W';
    if iterW==1
        tolW = 0.1 * tolW; 
    end
    
    [H,gradH,iterH] = pnm_nlssubprob(V,W,H,tolH,1000);
    if iterH==1
        tolH = 0.1 * tolH;
    end
    
    temp = toc;
    objstr = [objstr;.5*norm(V - W*H,'fro')^2];
    etime = etime + toc - temp;
    timeaxis = [timeaxis,toc - etime];
    gradW = W*(H*H') - V*H';
    projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
end
time = toc - etime;
fprintf('\nFinal Iteration %d\n', iter);
end


function [H,GRAD,iter] = pnm_nlssubprob(V,W,Hinit,tol,maxiter)
% Solve nonnegative least squares subproblem:
% min_H 1/2*||V - WH||_F^2  s.t. H_ij >= 0

% ----------------- INPUT ---------------------------
% V, W: nonnegative constant matrices
% Hinit: initial solution
% tol: stopping tolerance
% maxiter: limit of iterations

% ----------------OUTPUT ---------------------------
% H: output solution 
% GRAD: output gradient
% iter: #iterations used

% Written by Pinghua Gong, Tsinghua University
% Part of this function is modified based on Prof. Chih-Jen Lin's nlssubprob.m


H = Hinit; 
WtV = W'*V;
WtW = W'*W;
[r,m] = size(H);

sigma = 1e-3; beta = 5e-1; epsilon = 1e-12;
for iter=1:maxiter
  GRAD = WtW*H - WtV;
  projgrad = norm(GRAD(GRAD < 0 | H >0));
  if projgrad < tol
    break;
  end
  
  epsilonk = min(epsilon,norm(H - max(0,H - GRAD),'fro'));
  INDk = ((H >= 0) & (H <= epsilonk) & (GRAD > 0));
  invINDk = xor(ones(r,m),INDk);
  Pk = computePk(WtW,GRAD,INDk,invINDk,r,m);
  
  % search step size
  mk = 0;
  for inner_iter=1:20
     Hn = max(0,H - beta^mk*Pk); dH = Hn - H;
     gradd=sum(sum(GRAD.*dH)); dQd = sum(sum((WtW*dH).*dH));
     if gradd + 0.5*dQd <= -sigma*(beta^mk*sum(sum(GRAD(invINDk).*Pk(invINDk))) - sum(sum(dH(INDk).*GRAD(INDk))))
        break;
     else
        mk = mk + 1;
     end 
  end
  H = Hn;
      
end

if iter==maxiter
  fprintf('Max iter in pnm_nlssubprob\n');
end
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