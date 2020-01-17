function [g] = full_loss_grad(X,y,w,lss,delta)
% d: dimension of feature
% X: d-by-1
% y: 1-by-1
% w: 1-by-s

% lss: loss function
  % 1: 'hinge'
  % 2: 'logistic' 
  % 3: 'least square'
  % 4: 'huber'
  % 5: 'squared hinge'
  % 6: 'hinge loss'
  % 7: 'absolute loss'
  
[d,n] =size(X);

%g = zeros(1,d);
g=0;
%% Now, g is not necessary d-dimension


if lss == 2 % 'logistic' 
  
   pred = -y.*(X'*w')';
   g1 = y;
   ind_1 = find(pred >= 37);
   ind_2 = find(pred<37&pred>-746);
   ind_3 = find(pred <= -746);
   g1(ind_1) = 1;
   Exp = exp(pred(ind_2));
   g1(ind_2) = (Exp./(1+Exp));
   g1(ind_3) = 0;
   g = -X*(y.*g1)'/n;
   g = g';
 
   %% when pred >= 37, exp(pred)/(1+exp(pred)) = 1
      %% when pred <= -746, exp(pred)/(1+exp(pred)) = 0
end

if lss == 3 % 'least'
  
   pred = w*X - y;
   g = pred * X'/n;
   
end


if lss == 5 % 'squared hinge'
   
   pred = (w*X).*y;
   pred = 2 * (1 - pred);
   pred(pred <= 0) = 0;
   size(pred);
   g = -y.* pred;
   g = g * X'/n;
   size(g);
end

if lss == 6 % 'hinge loss'
 
  % size(y)
   pred = (w*X).*y;
   %size(pred)
   y = y*ones(1, n);
   y(pred >= 1) = 0;
  % size(y)
   g = - y * X'/n;
  % size(g)
end

if lss == 7 % 'absolute loss'
  
    pred = w*X - y;
    %[row, col] = size(pred);
    %disp(sprintf('full gradient pred: row = %d | col = %d | n = %d' , row, col, n));
    pred(pred>0) = 1;
    pred(pred < 0) = -1;
    %[row, col] = size(pred);
    %pred
    %disp(sprintf('full gradient pred: row = %d | col = %d | n = %d', row, col, n));
    g = pred * X'/n;
end





