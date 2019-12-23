[x_train, y_train, x_val, y_val, tr_reg, val_reg] = HW4_Utils.getPosAndRandomNeg();

x_train = x_train';
x_val = x_val';
C = 0.1;

[alpha, obj] = qp(x_train, y_train, C);
[alpha_w, alpha_h] = size(alpha);
[n,d] = size(x_train);
w = zeros(1, d);
b = 0;
n_vec = 0;
w  = x_train' * (y_train .* alpha);
[min_alpha, i] = min(abs(alpha));
b = trLb(i) - (w' * x_train(i,:)');

HW4_Utils.genRsltFile(w,b, 'val', 'res.mat');
[ap, prec, rec] = HW4_Utils.cmpAP('res.mat', 'val');

function [alpha, obj] = qp(X, y, C)
H = (y * y') .* (X * X');
[n,d] = size(X);
f = -1*ones(n,1);
A = [];
b = [];
Aeq = y';
beq  = 0;
lb = zeros(n,1);
ub = C * ones(n,1);

[alpha,obj] = quadprog(H, f, A, b, Aeq, beq);

end
