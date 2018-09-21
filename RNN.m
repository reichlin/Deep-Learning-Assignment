function assignment4()

    % read data
    book_fname = 'goblet_book.txt';
    fid = fopen(book_fname, 'r');
    book_data = fscanf(fid, '%c'); 
    fclose(fid);
    book_chars = unique(book_data);
    K = length(book_chars);

    char_to_ind = containers.Map(num2cell(book_chars), 1:K);
    ind_to_char = containers.Map(1:K, num2cell(book_chars));
    
    % parameters
    booklength = length(book_data);
    m = 100; % 100  % hidden state dimension
    seq_length = 25;
    sig = .1;
    GDparams.n_epoch = 20;
    GDparams.eta = 0.1;
    GDparams.epsilon = 1e-8;


    RNN.b = zeros(m, 1);
    RNN.c = zeros(K, 1);

    RNN.U = randn(m, K)*sig;
    RNN.W = randn(m, m)*sig;
    RNN.V = randn(K, m)*sig;

    X_hot = zeros(K, seq_length);
    Y_hot = zeros(K, seq_length);

    X = book_data(1:seq_length);
    Y = book_data(2:seq_length+1);
    for i=1:seq_length
        X_hot(char_to_ind(X(i)), i) = 1;
        Y_hot(char_to_ind(Y(i)), i) = 1;
    end

%     grads = ComputeGradients(RNN, seq_length, X_hot, Y_hot, zeros(size(RNN.b)));
%     Gs = ComputeGradsNum(X_hot, Y_hot, RNN, 1e-4);
% 
%     errorW = sqrt(sum(sum(sum((grads.W - Gs.W).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.W).^2))))+sqrt(sum(sum(sum((Gs.W).^2)))))
%     errorU = sqrt(sum(sum(sum((grads.U - Gs.U).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.U).^2))))+sqrt(sum(sum(sum((Gs.U).^2)))))
%     errorV = sqrt(sum(sum(sum((grads.V - Gs.V).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.V).^2))))+sqrt(sum(sum(sum((Gs.V).^2)))))
%     errorb = sqrt(sum(sum(sum((grads.b - Gs.b).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.b).^2))))+sqrt(sum(sum(sum((Gs.b).^2)))))
%     errorc = sqrt(sum(sum(sum((grads.c - Gs.c).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.c).^2))))+sqrt(sum(sum(sum((Gs.c).^2)))))

    [RNNstar, h] = AdaGrad(booklength, seq_length, RNN, GDparams, book_data, char_to_ind, ind_to_char, K);

    yy = sintethize(RNNstar, h, X_hot(:, 1), 1000);

    text = [];
    for w=1:1000
        text = [text, ind_to_char(find(yy(:, w)))];
    end
    text


end

function P = softmax(s)

    P = exp(s);
    for i=1:size(s, 2)
        P(:, i) = P(:, i) ./ sum(P(:, i));
    end
end

function Y = sintethize(RNN, h0, x0, N)

    Y = zeros(size(RNN.c, 1), N);
    h = h0;
    x = x0;

    for n=1:N
        a = RNN.W * h + RNN.U * x + RNN.b;
        h = tanh(a);
        o = RNN.V * h + RNN.c;
        p = softmax(o);
        
        % sample character
        cp = cumsum(p);
        a = rand;
        ixs = find(cp-a > 0);
        ii = ixs(1);
        
        Y(ii, n) = 1;
        
        x = Y(:, n);
    end
end

function L = ComputeLoss(X, Y, RNN, hprev)

    [P, ~, ~] = forward(RNN, X, hprev);
    
    N = size(P,2);

    L = 0;
    for i=1:N
        L = L - log(Y(:, i)' * P(:, i));
    end
end

function [P, H, A] = forward(RNN, X_chars, h)

    N = size(X_chars, 2);
    
    P = zeros(size(RNN.c, 1), N);
    H = zeros(size(RNN.b, 1), N+1);
    A = zeros(size(RNN.b, 1), N);
    
    H(:, 1) = h;
    for n=1:N
        A(:, n) = RNN.W * h + RNN.U * X_chars(:, n) + RNN.b;
        h = tanh(A(:, n));
        H(:, n+1) = h;
        o = RNN.V * h + RNN.c;
        P(:, n) = softmax(o);
    end
end

function [grads, h, L] = ComputeGradients(RNN, seq_length, X, Y, hprev)

    [P, H, A] = forward(RNN, X, hprev);
    
    grads.V = zeros(size(RNN.V));
    grads.W = zeros(size(RNN.W));
    grads.U = zeros(size(RNN.U));
    grads.b = zeros(size(RNN.b));
    grads.c = zeros(size(RNN.c));
    
    grad_a = zeros(1, size(RNN.W, 1));
    for t=seq_length:-1:1
        
        g = -(Y(:, t) - P(:, t))';
        grads.V = grads.V + g' * H(:, t+1)'; 
        grads.c = grads.c + g';
        
        grad_h = g * RNN.V + grad_a * RNN.W; 
        grad_a = grad_h * diag(1 - tanh(A(:, t)).^2);
        grads.W = grads.W + grad_a' * H(:, t)';
        grads.U = grads.U + grad_a' * X(:, t)';
        grads.b = grads.b + grad_a';
        
    end    
    
    for f = fieldnames(grads)'
       grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
    end
    
    h = H(:, seq_length+1);
    
    N = size(P,2);

    L = 0;
    for i=1:N
        L = L - log(Y(:, i)' * P(:, i));
    end
    
end

function [RNNstar, h] = AdaGrad(booklength, seq_length, RNN, GDparams, book_data, char_to_ind, ind_to_char, K)
    
    ages = floor(booklength/seq_length);
    
    mV = zeros(size(RNN.V));
    mW = zeros(size(RNN.W));
    mU = zeros(size(RNN.U));
    mb = zeros(size(RNN.b));
    mc = zeros(size(RNN.c));
    
    RNNstar.V = RNN.V;
    RNNstar.W = RNN.W;
    RNNstar.U = RNN.U;
    RNNstar.b = RNN.b;
    RNNstar.c = RNN.c;
    
    smooth_loss = 0;
    
    yloss = [];
    x = 0;
    
    minloss = 1e6;
    
    for i=1:GDparams.n_epoch
        
        e = 1; % keeps track of where in the book you are.
        hprev = zeros(size(RNN.b));
        
        for j=1:ages
            
            X_hot = zeros(K, seq_length);
            Y_hot = zeros(K, seq_length);

            X = book_data(e:e+seq_length-1);
            Y = book_data(e+1:e+seq_length);
            for s=1:seq_length
                X_hot(char_to_ind(X(s)), s) = 1;
                Y_hot(char_to_ind(Y(s)), s) = 1;
            end
            
            if smooth_loss == 0
                smooth_loss = ComputeLoss(X_hot, Y_hot, RNN, hprev);
                
                Y = sintethize(RNN, hprev, X_hot(:, 1), 200);

                text = [];
                for w=1:200
                    text = [text, ind_to_char(find(Y(:, w)))];
                end
                text
            end
            
            
            [grads, hprev, loss] = ComputeGradients(RNN, seq_length, X_hot, Y_hot, hprev);
            
            mV = mV + grads.V.^2;
            mW = mW + grads.W.^2;
            mU = mU + grads.U.^2;
            mb = mb + grads.b.^2;
            mc = mc + grads.c.^2;
            
            RNN.V = RNN.V - (GDparams.eta./sqrt(mV + GDparams.epsilon)).*grads.V;
            RNN.W = RNN.W - (GDparams.eta./sqrt(mW + GDparams.epsilon)).*grads.W;
            RNN.U = RNN.U - (GDparams.eta./sqrt(mU + GDparams.epsilon)).*grads.U;
            RNN.b = RNN.b - (GDparams.eta./sqrt(mb + GDparams.epsilon)).*grads.b;
            RNN.c = RNN.c - (GDparams.eta./sqrt(mc + GDparams.epsilon)).*grads.c;
            
            smooth_loss = .999 * smooth_loss + .001 * loss;
            
            if mod(j+i*ages, 100) == 0
                yloss = [yloss, smooth_loss];
                x = x + 1;
            end
            
            
            if mod(j+i*ages, 10000) == 0
                smooth_loss
                
                
                if minloss > smooth_loss
                    minloss = smooth_loss;
                    RNNstar = RNN;
                end
                
                Y = sintethize(RNN, hprev, X_hot(:, 1), 200);
    
                text = [];
                for w=1:200
                    text = [text, ind_to_char(find(Y(:, w)))];
                end
                text
            end
            
            
            e = e + seq_length;
            
        end
    end
    
    h = hprev;
    
    plot(1:x, yloss), grid on
    title("loss")

end

function num_grads = ComputeGradsNum(X, Y, RNN, h)
    for f = fieldnames(RNN)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, RNN, h);
    end
end

function grad = ComputeGradNum(X, Y, f, RNN, h)
    n = numel(RNN.(f));
    grad = zeros(size(RNN.(f)));
    hprev = zeros(size(RNN.W, 1), 1);
    for i=1:n
        RNN_try = RNN;
        RNN_try.(f)(i) = RNN.(f)(i) - h;
        l1 = ComputeLoss(X, Y, RNN_try, hprev);
        RNN_try.(f)(i) = RNN.(f)(i) + h;
        l2 = ComputeLoss(X, Y, RNN_try, hprev);
        grad(i) = (l2-l1)/(2*h);
    end

end
