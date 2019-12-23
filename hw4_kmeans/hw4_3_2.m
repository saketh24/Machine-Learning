[x_train, y_train, x_val, y_val, tr_reg, val_reg] = HW4_Utils.getPosAndRandomNeg();
x_train = x_train';
x_val = x_val';
C = 10;

[alpha, obj] = qp(x_train, y_train, C);
[alpha_w, alpha_h] = size(alpha);
[n,d] = size(x_train);
w = zeros(1, d);
w  = x_train' * (y_train .* alpha);
[min_alpha, i] = min(abs(alpha));
b = trLb(i) - (w' * x_train(i,:)');
obj_vals = [];
ap_vals = [];

for iter = 1:10
    x_t = [];
    y_t = [];
   disp(iter);
   PosD = [];
   NegD = [];
   for i = 1:size(y_train, 1)
       if y_train(i) == 1
           PosD = [PosD, x_train(i, :)'];
       else
           if alpha(i) > 0
               NegD = [NegD, x_train(i, :)'];
           end
       end
   end
   hard_neg = [];
   for i = 1:362
       im = imread(sprintf('%s/%sIms/%04d.jpg', HW4_Utils.dataDir, 'train', i));
       [ub1,ub2] = size(ubAnno{i});
       ub_i = ubAnno{i};
       rect = HW4_Utils.detect(im, w, b, 0);
       [imH, imW,~] = size(im);
       badIdxs = or(rect(3,:) > imW, rect(4,:) > imH);
       rect = rect(:,~badIdxs);
       for j = 1:length(rect)
           flag = 0;
           if rect(5,j) > 0
               continue
           end
           for k = 1:ub2
               if HW4_Utils.rectOverlap(rect, ub_i(:,k)) > 0.3
                   flag =1;
                   break;
               end
           if flag ==0
                imReg = im(int16(rect(2, j)):int16(rect(4, j)), int16(rect(1, j)):int16(rect(3, j)), :);
                imReg = imresize(imReg, HW4_Utils.normImSz);
                feat = HW4_Utils.cmpFeat(rgb2gray(imReg));
                feat = feat / norm(feat);
                hard_neg = [hard_neg, feat];     
                if size(hard_neg,2) > 1000
                    break;
                end
           end
           end
       end
       if size(hard_neg,2) > 1000
            break;
       end
   end
   NegD = [NegD, hard_neg];
   [neg_w, neg_h] = size(NegD);
   [pos_w, pos_h] = size(PosD);
   n_labels = -1 * ones(neg_h,1);
   p_labels = ones(pos_h,1);
   x_t = [x_t,PosD];
   y_t = [y_t;p_labels];
   x_t = [x_t, NegD];
   y_t = [y_t;n_labels];
   x_t = x_t';
   [alpha_new, obj_new] = qp(x_t, y_t, C);
   obj_vals = [obj_vals,obj_new];
   [x_w,x_h] = size(x_t);
   w_new = zeros(1, x_h);
   w_new  = x_t' * (y_t .* alpha_new);
   [min_alpha, i] = min(abs(alpha_new));
   b_new = y_t(i) - (w_new' * x_t(i,:)');
   HW4_Utils.genRsltFile(w_new, b_new, "val", "val_op.mat");
   [ap, prec, rec] = HW4_Utils.cmpAP("val_op.mat", "val");
   ap_vals = [ap_vals,ap];
   disp(ap);
   disp(obj);
end

function [alpha, obj] = qp(X, y, C)
H = (y * y') .* (X * X');
H = double(H);
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
