function [g, ind] = loss_grad(X,y,w,lss,delta)
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


ind = find(X);
%g = zeros(1,d);
g=0;
%% Now, g is not necessary d-dimension

if lss == 1 % 'hinge'
   x = X(ind);
   temp = y*x;
   pred = w(ind)*temp;
   if pred < 1
     %g(ind) = -temp';
     g = -temp';
   end
end

if lss == 2 % 'logistic' 
   x = X(ind);
   temp = -y*x;
   pred = w(ind)*temp;
      %% when pred >= 37, exp(pred)/(1+exp(pred)) = 1
      %% when pred <= -746, exp(pred)/(1+exp(pred)) = 0
   if pred > 37
      %g(ind) = temp'; 
      g = temp';
   elseif pred > -746
      Exp = exp(pred);
      %g(ind)  = (Exp/(1+Exp))*temp';
      g = (Exp/(1+Exp))*temp';
   end
end

if lss == 3 % 'least'
    x = X(ind);
    pred = w(ind)*x - y;
    %g(ind) = pred*x';
    g = pred*x';
end

if lss == 4  % 'huber'
   x = X(ind);
   pred = w(ind)*x - y;
   if abs(pred) <= delta
      %g(ind) = pred*x';
      g = pred*x'; 
   else
      if pred > 0
        %g(ind) = delta*x';
        g = delta*x';
      elseif pred < 0
        %g(ind) = -delta*x';
        g = -delta*x';
      end
   end
end

if lss == 5 % 'squared hinge'
   x = X(ind);
   temp = y*x;
   pred = w(ind)*temp;
   if pred < 1
     %g(ind) =2*(pred-1)*temp';
     g =2*(pred-1)*temp';
   end
end

if lss == 6 % 'hinge'
    x = X(ind);
    temp = y*x; 
    pred = w(ind)*temp;
    if pred < 1
      g = -temp';
    end
end

if lss == 7 % 'absolute'
    x = X(ind);
    pred = w(ind)*x - y;
    
    [row, col] = size(x);
    %disp(sprintf('loss gradient x: row = %d | col = %d', row, col));
    %[row, col] = size(w(ind));
    %disp(sprintf('loss gradient w: row = %d | col = %d', row, col));
    %[row, col] = size(pred);
    %disp(sprintf('loss gradient pred: row = %d | col = %d', row, col));
    if pred > 0
        g = x';
    elseif pred == 0
        g = 0;
    else
        g = -x';
    end
    %[row, col] = size(g);
    %disp(sprintf('loss gradient --g--: row = %d | col = %d', row, col));

end








