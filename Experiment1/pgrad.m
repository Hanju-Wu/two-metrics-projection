function [W,H,iter,time,timeaxis,objstr] = pgrad(V,Winit,Hinit,tol,timelimit,maxiter)

% NMF by projected gradients
% Author: Hanju Wu, Sun Yat-sen University
% Part of this function is modified based on Prof. Chih-Jen Lin's
% alspgrad.m and Dr. Yue Xie's pgrad.m

% W,H: output solution
% Winit,Hinit: initial solution
% tol: tolerance for a relative stopping condition
% timelimit, maxiter: limit of time and iterations

W = Winit; H = Hinit; tic; etime = 0;
E = W*H - V;
gradW = E*H'; gradH = W'*E;
objnow = .5*norm(E,'fro')^2;
tol = max(0.001,tol);%*initgrad; Yue changed this to absolute error
objstr = [];
timeaxis = [];
projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
for iter=1:maxiter
    % stopping condition
    if toc > timelimit || projnorm < tol %Yue changed the stopping criterion
        break;
    end

    for j = 1:50
        theta = .5^(j-1);
        Wn = max(W - theta*gradW,0); Hn = max(H - theta*gradH,0);
        En = Wn*Hn - V; objnew = .5*norm(En,'fro')^2;
        if objnew - objnow < .5*sum(sum([Wn' - W',Hn - H].*[gradW',gradH]))
            W = Wn; H = Hn; E = En; objnow = objnew;
            break;
        end
    end
    
    gradW = E*H'; gradH = W'*E;
    temp = toc;
    objstr = [objstr;.5*norm(W*H - V,'fro')^2];
    etime = etime + toc - temp;
    timeaxis = [timeaxis, toc - etime];
    projnorm = norm([gradW(gradW<0 | W>0); gradH(gradH<0 | H>0)]);
end
time = toc - etime;
fprintf('\nIter = %d Final proj-grad norm %f\n', iter, projnorm);
fprintf('Running time %fs\n', time);