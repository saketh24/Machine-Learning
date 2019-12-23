tr_data = csvread('train_data.csv',0,0);
tr_data = normalize(tr_data);
tr_label = csvread('tr_labels.csv',0,0);
val_data = csvread('Val_Data.csv',0,0);
val_data = normalize(val_data);
val_label = csvread('val_lab.csv',0,0);
w_vecs = [];
b_vals = [];
[c,d] = size(tr_label);
[a,b] = size(tr_data);
[n_val, d_val] = size(val_data);
t_labels = zeros(c,4);
C = 10;
for n = 1:4
for i = 1:c
    if tr_label(i) == n
        t_labels(i,n) = 1;
    else
        t_labels(i,n) = -1;
    end
end
end
for n = 1:4
   [alpha,obj] = qp(tr_data, t_labels(:,n), C);
   [alpha_w,alpha_h] = size(alpha);
   w = zeros(1,b);
   for i = 1:alpha_w
       w = w + (alpha(i) * t_labels(i,n) * tr_data(i,:));
   end
   b_val = 0;
   for i = 1:alpha_w
    if((alpha(i) > 0) && (alpha(i) < C))
        b_val = t_labels(i,n) - w * tr_data(i, :)';
        break
    end
   end
   b_vals = [b_vals;b_val];
   w_vecs = [w_vecs;w];
end
val_preds = zeros(1,n_val);
val_count = 0;
for i =1:n_val
    max = -10000;
    max_v = 0;
    for n = 1:4
        y_pred = val_data(i,:)*w_vecs(n,:)' + b_vals(n);
        if(y_pred > max)
            max = y_pred;
            max_v = n;
        end
    end
    val_preds(i) = max_v;
    if val_preds(i) == val_label(i)
        val_count = val_count + 1;
    end
end
disp(val_count/n_val);
function [alpha, obj] = qp(X, y, C)
H = (y * y') .* (X * X');
[n,d] = size(X);
f = -1* ones(n,1);
A = [];
b = [];
Aeq = y';
beq  = 0;
lb = zeros(n,1);
ub = C * ones(n,1);

[alpha,obj] = quadprog(H, f, A, b, Aeq, beq);

end