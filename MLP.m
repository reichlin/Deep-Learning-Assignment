function Assignment2bonus()

    GDparams.n_epochs = 100;
    GDparams.n_batch = 20;
    GDparams.eta_decay = 0.95;
    validation_size = 1000;
    GDparams.rho = 0.9;
    lambda = 0.000773750239891162;
    GDparams.eta = 0.0203625622751295;
    leaky = 0;
    alpha = 0.1;
    
    % load file
    [X1, Y1, y1] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_1.mat');
    [X2, Y2, y2] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_2.mat');
    [X3, Y3, y3] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_3.mat');
    [X4, Y4, y4] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_4.mat');
    [X5, Y5, y5] = LoadBatch('./Datasets/cifar-10-batches-mat/data_batch_5.mat');
    Xtrain = [X1, X2, X3, X4, X5(:, 1:(size(X5, 2)-validation_size))];
    Ytrain = [Y1, Y2, Y3, Y4, Y5(:, 1:(size(Y5, 2)-validation_size))];
    ytrain = [y1; y2; y3; y4; y5(1:(size(y5, 1)-validation_size), :)];
    Xvalidation = X5(:, (size(X5, 2)-validation_size+1):size(X5, 2));
    Yvalidation = Y5(:, (size(Y5, 2)-validation_size+1):size(Y5, 2));
    yvalidation = y5((size(y5, 1)-validation_size+1):size(y5, 1), :);
    
    clear X1 X2 X3 X4 X5 Y1 Y2 Y3 Y4 Y5 y1 y2 y3 y4 y5
    
    idx = randperm(2*size(Xtrain, 2));
    Xtrain = [Xtrain, jitter(Xtrain)];
    Ytrain = [Ytrain, Ytrain];
    ytrain = [ytrain; ytrain];
    Xtrain = Xtrain(:, idx);
    Ytrain = Ytrain(:, idx);
    ytrain = ytrain(idx);
    
    
    [Xtest, ~, ytest] = LoadBatch('./Datasets/cifar-10-batches-mat/test_batch.mat');
    
    mean_X = mean(Xtrain, 2);
    Xtrain = Xtrain - repmat(mean_X, [1, size(Xtrain, 2)]);
    Xvalidation = Xvalidation - repmat(mean_X, [1, size(Xvalidation, 2)]);
    Xtest = Xtest - repmat(mean_X, [1, size(Xtest, 2)]);
    
    Ntrain = size(Xtrain, 2);
    d = size(Xtrain, 1);
    K = 10;
    m = 80;
     
    rng(400); 
    
    [W1, b1] = NetworkInit(d, m, K);

    [W, b] = MiniBatchGD(Xtrain, Ytrain, ytrain, GDparams, W1, b1, lambda, Xvalidation, Yvalidation, yvalidation, leaky, alpha);

    acc = ComputeAccuracy(Xtest, ytest, W, b, leaky, alpha)

    
end


function [X, Y, y] = LoadBatch(filename)

    A = load(filename);
   
    N = size(A.data, 1);
    K = 10;
   
    X = double(A.data') / 255;
    y = double(A.labels) + ones(N, 1);
    
    Y = zeros(K, N);
    for i=1:N
        Y(y(i), i) = 1;
    end
end

function [W, b] = NetworkInit(d, m, K)
    
    b1 = zeros(m, 1);
    b2 = zeros(K, 1);
    
    % He initialization
    var = sqrt(2/d);
    
    W1 = randn(m, d).*var;
    W2 = randn(K, m).*var;
    
    W = {W1, W2};
    b = {b1, b2};
end
 

function P = EvaluateClassifier(X, W, b, leaky, alpha)

    if leaky == 0
        h = EvaluateHidden(X, W{1}, b{1});
    else
        h = EvaluateHiddenLeaky(X, W{1}, b{1}, alpha);
    end
    s = W{2} * h + repmat(b{2}, 1, size(h, 2));
    P = softmax(s);
end

function h = EvaluateHidden(X, W1, b1)
    
    s = W1 * X + repmat(b1, 1, size(X, 2));
    h = max(s, 0);
    
end

function h = EvaluateHiddenLeaky(X, W1, b1, alpha)
    
    s = W1 * X + repmat(b1, 1, size(X, 2));
    h = max(s, 0) + alpha .* min(s, 0);
    
end

function P = softmax(s)

    P = exp(s);
    for i=1:size(s, 2)
        P(:, i) = P(:, i) ./ sum(P(:, i));
    end
end

function J = ComputeCost(X, Y, W, b, lambda, leaky, alpha)

    N = size(X, 2);
    P = EvaluateClassifier(X, W, b, leaky, alpha);
   
    J = 0;
    for i=1:N
        J = J - log(Y(:, i)' * P(:, i));
    end
    J = J / N + lambda * sum(sum(W{1}.^2)) + lambda * sum(sum(W{2}.^2));
end

function J = ComputeLoss(X, Y, W, b, leaky, alpha)

    N = size(X, 2);
    P = EvaluateClassifier(X, W, b, leaky, alpha);
   
    J = 0;
    for i=1:N
        J = J - log(Y(:, i)' * P(:, i));
    end
    J = J / N;
end

function acc = ComputeAccuracy(X, y, W, b, leaky, alpha)

    N = size(X,2);
    [~, P] = max(EvaluateClassifier(X, W, b, leaky, alpha));
   
    acc = 100 * (sum(y == P') / N);
end

function [grad_W, grad_b] = ComputeGradients(X, Y, P, W, b, lambda, leaky, alpha)

    K = size(Y, 1);
    d = size(X, 1);
    N = size(X, 2);
    m = size(W{1}, 1);
    
    grad_W1 = zeros(m, d);
    grad_b1 = zeros(m, 1);
    grad_W2 = zeros(K, m);
    grad_b2 = zeros(K, 1);
    
    if leaky == 0
        h = EvaluateHidden(X, W{1}, b{1});
    else
        h = EvaluateHiddenLeaky(X, W{1}, b{1}, alpha);
    end
    
    for i=1:N
        g = -(Y(:, i) - P(:, i))';
        
        grad_W2 = grad_W2 + g'*h(:, i)';
        grad_b2 = grad_b2 + g';
        
        if leaky == 0
            g = g*W{2}*diag(h(:, i) > 0);
        else
            g = g*W{2}*(diag(h(:, i) > 0) + alpha.*diag(h(:, i) < 0));
        end
        grad_W1 = grad_W1 + g'*X(:, i)';
        grad_b1 = grad_b1 + g';
    end
    
    grad_W1 = grad_W1 ./ N + 2*lambda .* W{1};
    grad_b1 = grad_b1 ./ N;
    grad_W2 = grad_W2 ./ N + 2*lambda .* W{2};
    grad_b2 = grad_b2 ./ N;
    
    grad_W = {grad_W1, grad_W2};
    grad_b = {grad_b1, grad_b2};
    
end

function [Wstar, bstar] = MiniBatchGD(X, Y, y, GDparams, W, b, lambda, Xvalidation, Yvalidation, yvalidation, leaky, alpha)

    N = size(X, 2);
    
    v = {zeros(size(W{1})), zeros(size(b{1})), zeros(size(W{2})), zeros(size(b{2}))};
    
    bestacc = 0;
    
    for i=1:GDparams.n_epochs
        
        
        for j=1:N/GDparams.n_batch
            j_start = (j-1)*GDparams.n_batch + 1;
            j_end = j*GDparams.n_batch;
            Xbatch = X(:, j_start:j_end);
            Ybatch = Y(:, j_start:j_end);
            ybatch = y(j_start:j_end);
            
            P = EvaluateClassifier(Xbatch, W, b, leaky, alpha);
            [grad_W, grad_b] = ComputeGradients(Xbatch, Ybatch, P, W, b, lambda, leaky, alpha);
            
            v{1} = (GDparams.rho .* v{1} + GDparams.eta .* grad_W{1});
            v{2} = (GDparams.rho .* v{2} + GDparams.eta .* grad_b{1});
            v{3} = (GDparams.rho .* v{3} + GDparams.eta .* grad_W{2});
            v{4} = (GDparams.rho .* v{4} + GDparams.eta .* grad_b{2});

            W{1} = W{1} - v{1};
            b{1} = b{1} - v{2};
            W{2} = W{2} - v{3};
            b{2} = b{2} - v{4};
            
        end
        
        GDparams.eta = GDparams.eta * GDparams.eta_decay;
        
        losst(i) = ComputeLoss(X, Y, W, b, leaky, alpha);
        lossv(i) = ComputeLoss(Xvalidation, Yvalidation, W, b, leaky, alpha);
        costt(i) = ComputeCost(X, Y, W, b, lambda, leaky, alpha);
        costv(i) = ComputeCost(Xvalidation, Yvalidation, W, b, lambda, leaky, alpha);
        acc(i) = ComputeAccuracy(Xvalidation, yvalidation, W, b, leaky, alpha)
        
        if bestacc < acc(i)
            bestacc = acc(i);
            Wstar = W;
            bstar = b;
        end


    end
    
%     plot(1:GDparams.n_epochs, losst), grid on, hold on
%     plot(1:GDparams.n_epochs, lossv)
%     xlabel('Number of epochs')
%     ylabel('loss')
%     legend('training set', 'validation set')
%     
%     figure
%     plot(1:GDparams.n_epochs, costt), grid on, hold on
%     plot(1:GDparams.n_epochs, costv), grid on, hold on
%     xlabel('Number of epochs')
%     ylabel('cost')
%     legend('training set', 'validation set')
    
%     figure,
%     plot(1:GDparams.n_epochs, acc), grid on, hold on
%     xlabel('Number of epochs')
%     ylabel('accuracy on the validation set')

end

function X_new = jitter(X)
    N = size(X,2);
   
   X_new = zeros(3072, N);
 
    for n=1:N
        for r=0:2
            for i=0:31
                for j=0:31
                    X_new(r*1024 + i*32 + 1 + j, n) = X(r*1024 + i*32 + 32 - j, n);
                end
            end
        end
    end
 
end
