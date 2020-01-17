function [objres] = g_obj(X,y,lss,lambda,w,delta)
% In pu_learning, no regularizer.
% lss: loss function
  % 1: 'hinge'
  % 2: 'logistic' 
  % 3: 'least square'
  % 4: 'huber'
  % 5: 'squared hinge'
  % 6: 'hinge loss'
  % 7: 'absolute loss'

 pred=w*X;

 if lss == 2 % 'logistic'
    objres = -y.*pred;
    id = find(objres <= 709); % 709.775 is safe    --  a note
    objres(id) = log(1+exp(objres(id)));
 end
 
 
 if lss == 3 % 'least square'
    objres =  0.5*(pred - y).^2;
 end

 if lss == 5 % 'squared hinge'
    objres = max(0, 1 - pred.* y);
    objres = objres.^2;	
 end
 
 if lss == 6 % ' hinge'
    objres = max(0, 1 - pred.* y);
 end
 
  if lss == 7 % ' absolute loss'
    objres = abs(pred - y);
 end

objres = mean(objres);
