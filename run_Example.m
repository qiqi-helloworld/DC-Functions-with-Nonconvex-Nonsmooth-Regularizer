
[y,X]=libsvmread('./data/real-sim');
X = X'; 
y = y';
[d,n]=size(X);
n_p = sum(y == 1);
pi_p = n_p/n;
X_p = X(:,y == 1);
rng(77);
ind_p = randsample(n_p, 1000, true);
X_p = X_p(:,ind_p);
X_u = X;

regs=1; K = 1e7; w0 = zeros(1,d);
NG = 11e6; 
lambda = 1e-4; theta = 0.1*lambda;
delta=1; epsilon=1; dense=1; evalk=200000;
idx_p=randsample(1000,NG,true);
idx_u=randsample(n, NG, true);
idx_n=randsample(1000,NG,true);
prox = 0; D = 1000; SVRGNG = 4.5e6;
last = 1;

%absolute loss
lss = 7;


disp(sprintf('SSPG'));evalk = 10;
gamm = 1e-2; T0 = 100; eta0 = 100;
disp(sprintf('SSDC_SGD | lss = %d | eta0  = %d | gamm = %d', lss, eta0, gamm));
[SPGObj, SPGit, SPGavgw]=SSDC_SGD(last, X_p, X_u, pi_p, lss, regs, eta0, T0, gamm, w0, K,lambda, theta, epsilon, delta, dense,prox,idx_p, idx_u, idx_n, evalk, D);
filename = sprintf('regs-morerun-SSPG-eta0-%.5f-lambda-%.d-T0-%d-gamm-%.5f-lss-%d-last-%d.csv', eta0,lambda, T0, gamm, lss, last);
csv_path = './realsim_csv_res/';
csvwrite(strcat(csv_path,filename),SPGObj);
   


evalk = 1;G = 10;eta0 = 100;T0 = 0.1;gamm = 1;
if gamm == 1e-2
    evalk = 1;
 elseif gamm >= 1
   evalk = 500;
end
disp(sprintf('SSDC_AdaGrad | lss = %d | gamm = %.0d | eta0 - %.5f | T = %d  ', lss, gamm, eta0, T0));
[AdaObj,Adait,Adaavgw]=SSDC_AdaGrad(last, X_p, X_u, pi_p, lss, regs, eta0, T0, gamm, G,  w0, K, lambda, theta,...
                             epsilon, delta, dense,prox,idx_p, idx_u, idx_n, evalk, D);
filename = sprintf('regs-morerun-SAdaSPG-eta0-%.5f-lambda-%.d-T0-%.d_gamm-%.0d-G-%.2f-lss-%d-last-%d.csv', eta0, lambda, T0, gamm, G, lss, last);
csv_path = './realsim_csv_res/';
csvwrite(strcat(csv_path,filename),AdaObj);

  
