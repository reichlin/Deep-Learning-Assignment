function Assignment3()

    % read data
    
    
    [all_names, ys] = ExtractNames('ascii_names.txt');
    
    rng(100);

    C = unique(cell2mat(all_names)); % all possible characters
    d = numel(C); % number of all possible characters

    N = length(all_names); % number of names

    % maximum length of names
    n_len = 0;
    for i=1:N
        if n_len < length(all_names{i})
            n_len = length(all_names{i});
        end
    end

    % number of classes
    K = length(unique(ys));

    % map characters to values
    char_to_ind = containers.Map(num2cell(C), 1:d);

    % encode names as one hot and fill matrix X
    X = zeros(n_len*d, N);
    word_1hot = zeros(d, n_len);
    for i=1:N
        word = all_names{i};
        for j=1:n_len
            letter = zeros(d, 1);
            if j <= length(word)
                letter(char_to_ind(word(j)), 1) = 1;
            end
            word_1hot(:, j) = letter;
        end
        X(:, i) = reshape(word_1hot, [n_len*d, 1]);
    end

    % indeces for the validation set
    val_ind = dlmread("Validation_Inds.txt");
    val_ind = val_ind(1:length(val_ind)-3);

    train_ind = zeros(1, length(N-val_ind));
    j=1;
    for i=1:N
        if ~sum(ismember(val_ind, i))
            train_ind(1,j) = i;
            j = j + 1;
        end
    end
    
    % parameters
    balanceDataset = 1;
    if balanceDataset == 1
        GDparams.n_epochs = 3000;
    else
        GDparams.n_epochs = 200;
    end
    GDparams.n_batch = 100;
    GDparams.eta = 0.001;
    GDparams.eta_decay = 1;
    GDparams.rho = 0.9;
    n1 = 30;
    k1 = 5;
    n2 = 20;
    k2 = 3;
    fsize = n2*(n_len-k1+1-k2+1);
    sig1 = sqrt(2/k1);
    sig2 = sqrt(2/(n1*k2));
    sig3 = sqrt(2/(fsize));
    convNet.F{1} = randn(d, k1, n1)*sig1;
    convNet.F{2} = randn(n1, k2, n2)*sig2;
    convNet.W = randn(K, fsize)*sig3;
    nlen1 = n_len;
    nlen2 = n_len-k1+1;

    % definition of training and validation set
    x_train = X(:, train_ind);
    y_train = ys(train_ind);
    x_val = X(:, val_ind);
    y_val = ys(val_ind);
    
    if balanceDataset ~= 1
        idx = randperm(size(x_train, 2));
        x_train = x_train(:, idx);
        y_train = y_train(idx);
    end
    
    % one-shot labels
    Y_train = zeros(K, size(y_train, 1));
    for i=1:size(y_train, 1)
        Y_train(y_train(i), i) = 1;
    end
    Y_val = zeros(K, size(y_val, 1));
    for i=1:size(y_val, 1)
        Y_val(y_val(i), i) = 1;
    end
    
    % precompute Mx for efficiency
    for i=1:size(x_train, 2)
        mx{i} = sparse(MakeMXMatrix(reshape(x_train(:,i), d, []), d, k1, n1));
    end
    
%     x_batch = x_train(:, 1:GDparams.n_batch);
%     Y_batch = Y_train(:, 1:GDparams.n_batch);
%     
%     Gs = NumericalGradient(x_batch, Y_batch, convNet, 1e-6);
%     [grad_F1, grad_F2, grad_W, mf1, mf2] = ComputeGradients(x_batch, Y_batch, convNet, GDparams.n_batch, nlen1, nlen2, mx(1:GDparams.n_batch));
%     
%     
%     errorf1 = sqrt(sum(sum(sum((grad_F1 - Gs{1}).^2)))) / max(1e-6, sqrt(sum(sum(sum((grad_F1).^2))))+sqrt(sum(sum(sum((Gs{1}).^2)))))
%     errorf2 = sqrt(sum(sum(sum((grad_F2 - Gs{2}).^2)))) / max(1e-6, sqrt(sum(sum(sum((grad_F2).^2))))+sqrt(sum(sum(sum((Gs{2}).^2)))))
%     errorw = sqrt(sum(sum(sum((grad_W - Gs{3}).^2)))) / max(1e-6, sqrt(sum(sum(sum((grad_W).^2))))+sqrt(sum(sum(sum((Gs{3}).^2)))))
    
    
    nMinorClass = sum(y_train == 1);
    for i=2:K
        if nMinorClass > sum(y_train == i)
            nMinorClass = sum(y_train == i);
        end
    end
    
    [all_names_test, ys_test] = ExtractNames('test.txt');
    
    % encode names as one hot and fill matrix X
    X_test = zeros(n_len*d, 6);
    word_1hot = zeros(d, n_len);
    for i=1:6
        word = all_names_test{i};
        for j=1:n_len
            letter = zeros(d, 1);
            if j <= length(word)
                letter(char_to_ind(word(j)), 1) = 1;
            end
            word_1hot(:, j) = letter;
        end
        X_test(:, i) = reshape(word_1hot, [n_len*d, 1]);
    end
    
    Y_test = zeros(K, size(ys_test, 1));
    for i=1:size(ys_test, 1)
        Y_test(ys_test(i), i) = 1;
    end
    
    convNet_star = MiniBatchGD(x_train, Y_train, x_val, Y_val, convNet, GDparams, nlen1, nlen2, mx, balanceDataset, nMinorClass, K);
    
    acc = ComputeAccuracy(X_test, Y_test, convNet_star, nlen1, nlen2)
    
end

function MF = MakeMFMatrix(F, nlen)

    [dd, k, nf] = size(F);
    
    MF_flat = reshape(F, [dd*k, nf])';
    
    MF_flat = [repmat([MF_flat, zeros(nf, dd*(nlen-k+1))], [1, (nlen-k+1)-1]), MF_flat];
    
    MF = MF_flat(:, 1:(dd*nlen));
    for i=1:(nlen-k)
        MF = [MF; MF_flat(:, (i*(dd*nlen)+1):(i+1)*(dd*nlen))];
    end

end

function MX = MakeMXMatrix(x_input, d, k, nf)

    nlen = size(x_input, 2);
    
    MX = zeros((nlen-k+1)*nf, d*k*nf);
    for i=1:(nlen-k+1)
        MX( (i-1)*nf+1:i*nf, : ) = kron(eye(nf), reshape(x_input(:, i:(i+k-1)), 1, []));
    end
    
end

function MX = MakeMXMatrixGeneralization(x_input, d, k)

    nlen = size(x_input, 2);
    
    MX = zeros((nlen-k+1), d*k);
    for i=1:(nlen-k+1)
        MX( i, : ) = reshape(x_input(:, i:(i+k-1)), 1, []);
    end
    
end

function [s, MF] = convLayer(x, F, nlen)

    MF = MakeMFMatrix(F, nlen);
    s = max(0, MF * x);
end

function p = denseLayer(x, W)

    s = W * x;
    p = softMax(s);
end

function p = softMax(x)

    p = exp(x);
    for i=1:size(x, 2)
        p(:, i) = p(:, i) ./ sum(p(:, i));
    end
end

function p = forward(x, convNet, nlen1, nlen2)

    s1 = convLayer(x, convNet.F{1}, nlen1);
    s2 = convLayer(s1, convNet.F{2}, nlen2);
    p = denseLayer(s2, convNet.W);
end

function J = ComputeLoss(X, Y, MFs, W)

    N = size(X, 2);
    P = denseLayer(max(0, MFs{2}*max(0, MFs{1}*X)), W);
   
    J = 0;
    for i=1:N
        J = J - log(Y(:, i)' * P(:, i));
    end
    J = J / N;
end

function acc = ComputeAccuracy(X, Y, convNet, nlen1, nlen2)

    N = size(X,2);
    P_tot = forward(X, convNet, nlen1, nlen2);
    [~, P] = max(P_tot);
    
    if N == 6
        P_tot
    end
   
    for i=1:N
        y(i) = find(Y(:, i));
    end
    acc = 100 * (sum(y == P) / N);
end

function confusionMatrix(P, Y)

    K = size(P, 1);
    M = zeros(K, K);

    for y=1:size(Y, 2)
        ind = find(Y(:, y));
        [~, i] = max(P(:, y));
        M(ind, i) = M(ind, i) + 1;
    end
    
    M ./ sum(M, 2)

end

function [grad_F1, grad_F2, grad_W, mf1, mf2] = ComputeGradients(x, Y, convNet, batch_size, nlen1, nlen2, mx1)

    [X1, mf1] = convLayer(x, convNet.F{1}, nlen1);
    [X2, mf2] = convLayer(X1, convNet.F{2}, nlen2);
    P = denseLayer(X2, convNet.W);
    
    [h1, w1, n1] = size(convNet.F{1});
    [h2, w2, n2] = size(convNet.F{2});
    
    grad_F1 = zeros(1, h1 * w1 * n1);
    grad_F2 = zeros(1, h2 * w2 * n2);
    
    g = -(Y - P);
    
    grad_W = (g * X2')/batch_size;
    
    g = convNet.W' * g;
    
    g = g .* (X2 > 0);
    
    for j=1:batch_size
        v = MakeMXMatrixGeneralization(reshape(X1(:,j), h2, []), h2, w2)' * reshape( g(:,j), [], (nlen2-w2+1) )';
        grad_F2 = grad_F2 + v(:)' ./ batch_size;
    end
    
    g = mf2' * g;
    g = g .* (X1 > 0);
    
    for j=1:batch_size
        v = g(:,j)' * mx1{j};
        grad_F1 = grad_F1 + v ./ batch_size;
    end
    
    grad_F1 = reshape(grad_F1, h1, w1, n1);
    grad_F2 = reshape(grad_F2, h2, w2, n2);
    
end

function convNet_star = MiniBatchGD(X_train_ordered, Y_train_ordered, X_val, Y_val, convNet, GDparams, nlen1, nlen2, mx_ordered, balanceDataset, nMinorClass, K)

    convNet_star.F{1} = convNet.F{1};
    convNet_star.F{2} = convNet.F{2};
    convNet_star.W = convNet.W;
    
    N = size(X_train_ordered, 2);
    
    v = {zeros(size(convNet.F{1})), zeros(size(convNet.F{2})), zeros(size(convNet.W))};
    
    bestacc = 0;
    
    n_ages = floor(N/GDparams.n_batch);
    
    counter = 1;
    
    if balanceDataset ~= 1
        X_train = X_train_ordered;
        Y_train = Y_train_ordered;
        mx = mx_ordered;
    end
    
    for i=1:GDparams.n_epochs
        
        if balanceDataset == 1
            [r, ~] = find(Y_train_ordered);
            nclass = sum(r == 1);
            idx = ceil(rand(1, nMinorClass(1)) .* nclass);
            
            cumNclass = nclass;
            
            for c=2:K
                [r, ~] = find(Y_train_ordered);
                nclass = sum(r == c);
                idx = [idx, ceil(rand(1, nMinorClass) .* nclass + (cumNclass))];
                cumNclass = cumNclass + nclass;
            end
            
            idx = idx(randperm(size(idx, 2)));
            
            X_train = X_train_ordered(:, idx);
            Y_train = Y_train_ordered(:, idx);
            
            mx = mx_ordered(idx);
            
            n_ages = floor(size(X_train, 2)/GDparams.n_batch);
        end
        
        for j=1:n_ages
            j_start = (j-1)*GDparams.n_batch + 1;
            j_end = j*GDparams.n_batch;
            Xbatch = X_train(:, j_start:j_end);
            Ybatch = Y_train(:, j_start:j_end);
            
            
            [grad_F1, grad_F2, grad_W, mf1, mf2] = ComputeGradients(Xbatch, Ybatch, convNet, GDparams.n_batch, nlen1, nlen2, mx(j_start:j_end));
            
            
            v{1} = (GDparams.rho .* v{1} + GDparams.eta .* grad_F1);
            v{2} = (GDparams.rho .* v{2} + GDparams.eta .* grad_F2);
            v{3} = (GDparams.rho .* v{3} + GDparams.eta .* grad_W);

            convNet.F{1} = convNet.F{1} - v{1};
            convNet.F{2} = convNet.F{2} - v{2};
            convNet.W = convNet.W - v{3};
            
            
            
            if mod(j+n_ages*i, 500) == 0
                MFs{1} = mf1;
                MFs{2} = mf2;
                val_loss(counter) = ComputeLoss(X_val, Y_val, MFs, convNet.W);
                acc_train(counter) = ComputeAccuracy(Xbatch, Ybatch, convNet, nlen1, nlen2);
                acc_val(counter) = ComputeAccuracy(X_val, Y_val, convNet, nlen1, nlen2);
                
                if bestacc < acc_val(counter)
                    bestacc = acc_val(counter);
                    convNet_star.F{1} = convNet.F{1};
                    convNet_star.F{2} = convNet.F{2};
                    convNet_star.W = convNet.W;
                end
                
                counter = counter + 1;
            end
            
            
        end
        
        GDparams.eta = GDparams.eta * GDparams.eta_decay;

    end
    
    counter = counter - 1;
    
    P_val = forward(X_val, convNet, nlen1, nlen2);
    confusionMatrix(P_val, Y_val);
    
    figure, plot(1:counter, val_loss), grid on
    title('validation loss')
    figure, plot(1:counter, acc_train), grid on, hold on
    plot(1:counter, acc_val)
    title('accuracy')
    legend('training accuracy', 'validation accuracy')

end





