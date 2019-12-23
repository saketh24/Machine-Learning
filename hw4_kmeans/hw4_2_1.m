load('q2_1_data.mat');
C = 0.1;
trD = trD';
valD = valD';
[alpha, obj] = qp(trD, trLb, C);
[alpha_w, alpha_h] = size(alpha);
[n,d] = size(trD);
[n_val, d_val] = size(valD);
w = zeros(1,d);
n_vec = 0;
w  = trD' * (trLb .* alpha);
[min_alpha, i] = min(abs(alpha));
b = trLb(i) - (w' * trD(i,:)');
for i = 1:alpha_w
    if alpha(i) > 0
        n_vec = n_vec + 1;
    end
end
% for i = 1:alpha_w
%     w = w + (alpha(i) * trLb(i) * trD(i,:)); 
%     if(alpha(i) > 0)
%         n_vec = n_vec + 1;
%     end
% end
% b = 0;
% for i = 1:alpha_w
%     if((alpha(i) > 0) && (alpha(i) < C))
%         b = trLb(i) - w * trD(i, :)';
%     end
% end
val_pred = zeros(1,n_val);
val_pred = (w' * valD') + b;
% for i = 1:n_val
%     if(valD(i,:) * w' + b > 0)
%         val_pred(i) = 1;
%     else
%         val_pred(i) = -1;
%     end   
% end
correct_count = 0;
for i = 1:n_val
    if(val_pred(i) > 0)
        val_pred(i) = 1;
    else
        val_pred(i) = -1;
    end
end

for i = 1:n_val
    if(val_pred(i) == valLb(i))
        correct_count = correct_count + 1;
    end
end

accuracy = (correct_count/ n_val);
disp(accuracy);
disp(n_vec);
disp(obj);
C = confusionmat(valLb,val_pred);
disp(C);
confusionchart(C);
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
