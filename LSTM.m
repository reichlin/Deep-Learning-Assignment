function LSTM()
    % READ TWEETS AND FIND UNIQUE CHARACTER
    tweets = readTweet();
    N = length(tweets);
    str = [];
    for i=1:N
        str = [str, tweets{i}];
    end
    chars = unique(str);
    d = length(chars);
    
    % MAP CHARS TO NUMBERS AND VICE VERSA
    char_to_ind = containers.Map(num2cell(chars), 1:d);
    ind_to_char = containers.Map(1:d, num2cell(chars));
    
    d = d + 1;
    
%     tau = 80; %length(tweets{1})-1;
    m = 100;
    h0 = zeros(m, 1);
    c0 = zeros(m, 1);
    CELL.W = randn(4*m, m).*0.01;
    CELL.U = randn(4*m, d).*0.01;
    CELL.V = randn(d, m).*0.01;
    CELL.b = zeros(4*m, 1);
    CELL.c = zeros(d, 1);
    
    GDparams.n_epochs = 1;
    GDparams.eta = 0.1;
    GDparams.epsilon = 1e-8;
    
%     X = zeros(d, tau);
%     Y = zeros(d, tau);
%     
%     for i=1:tau
%         ind = char_to_ind(tweets{1}(i));
%         ind2 = char_to_ind(tweets{1}(i+1));
%         X(ind, i) = 1;
%         Y(ind2, i) = 1;
%     end
%     
%     
%     [grads, h, c, loss] = backpropagation(CELL, h0, c0, X, Y, d, m, tau);
%     
%     Gs = ComputeGradsNum(X, Y, CELL, 1e-4);
% 
%     errorW = sqrt(sum(sum(sum((grads.W - Gs.W).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.W).^2))))+sqrt(sum(sum(sum((Gs.W).^2)))))
%     errorU = sqrt(sum(sum(sum((grads.U - Gs.U).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.U).^2))))+sqrt(sum(sum(sum((Gs.U).^2)))))
%     errorV = sqrt(sum(sum(sum((grads.V - Gs.V).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.V).^2))))+sqrt(sum(sum(sum((Gs.V).^2)))))
%     errorb = sqrt(sum(sum(sum((grads.b - Gs.b).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.b).^2))))+sqrt(sum(sum(sum((Gs.b).^2)))))
%     errorc = sqrt(sum(sum(sum((grads.c - Gs.c).^2)))) / max(1e-6, sqrt(sum(sum(sum((grads.c).^2))))+sqrt(sum(sum(sum((Gs.c).^2)))))


    CELLstar = AdaGrad(tweets, CELL, GDparams, ind_to_char, char_to_ind, d, m);
    
    x0 = zeros(d, 1);
    x0(70) = 1;

    synthesize(CELLstar, h0, c0, x0, d, m, ind_to_char)
    
end

function tweets = readTweet()
    n = 1;
    jsonData = jsondecode(fileread('Tweets/condensed_2009.json'));
    for i=1:length(jsonData)
        tweets{i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2010.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2011.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2012.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2013.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2014.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2015.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2016.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2017.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
    n = n + length(jsonData);
    jsonData = jsondecode(fileread('Tweets/condensed_2018.json'));
    for i=1:length(jsonData)
        tweets{n+i} = jsonData(i).text;
    end
end

function str = synthesize(CELL, h0, c0, x0, d, m, ind_to_char)

    str = [];

    h = h0;
    c = c0;
    x = x0;
    E1 = [eye(m), zeros(m, m), zeros(m, m), zeros(m, m)];
    E2 = [zeros(m, m), eye(m), zeros(m, m), zeros(m, m)];
    E3 = [zeros(m, m), zeros(m, m), eye(m), zeros(m, m)];
    E4 = [zeros(m, m), zeros(m, m), zeros(m, m), eye(m)];
    
    flag = 1;
    char = 1;

    while flag == 1
        a = CELL.W * h + CELL.U * x + CELL.b;
        f = sigmoid(E1*a);
        i = sigmoid(E2*a);
        c_tilde = tanh(E3*a);
        o = sigmoid(E4*a);
        c = f .* c + i .* c_tilde;
        h = o .* tanh(c);
        out = CELL.V * h + CELL.c;
        
        p = softmax(out);
%         [~, ind] = max(p);
%         x = zeros(d, 1);
%         x(ind) = 1;
        
        cp = cumsum(p);
        ixs = find(cp-rand > 0);
        ind = ixs(1);
        x = zeros(d, 1);
        x(ind) = 1;
        
        if (ind == d || char == 140)
            flag = 0;
        else
            str = [str, ind_to_char(ind)];
            char = char + 1;
        end
    end

end

function [P, H, C, A, F, I, C_tilde, O] = forward(CELL, h0, c0, x, d, m, E1, E2, E3, E4)

    tau = size(x, 2);

    H = zeros(m, tau+1);
    C = zeros(m, tau+1);
    A = zeros(4*m, tau);
    F = zeros(m, tau);
    I = zeros(m, tau);
    C_tilde = zeros(m, tau);
    O = zeros(m, tau);
    P = zeros(d, tau);

    H(:, 1) = h0;
    C(:, 1) = c0;

    for t=1:tau
        A(:, t) = CELL.W * H(:, t) + CELL.U * x(:, t) + CELL.b;
        F(:, t) = sigmoid(E1*A(:, t));
        I(:, t) = sigmoid(E2*A(:, t));
        C_tilde(:, t) = tanh(E3*A(:, t));
        O(:, t) = sigmoid(E4*A(:, t));
        C(:, t+1) = F(:, t) .* C(:, t) + I(:, t) .* C_tilde(:, t);
        H(:, t+1) = O(:, t) .* tanh(C(:, t+1));
        out = CELL.V * H(:, t+1) + CELL.c;
        
        P(:, t) = softmax(out);
    end

end

function L = ComputeLoss(X, Y, CELL, h0, c0)
    
    tau = size(X,2);
    d = size(X, 1);
    m = size(CELL.V, 2);
    
    E1 = [eye(m), zeros(m, m), zeros(m, m), zeros(m, m)];
    E2 = [zeros(m, m), eye(m), zeros(m, m), zeros(m, m)];
    E3 = [zeros(m, m), zeros(m, m), eye(m), zeros(m, m)];
    E4 = [zeros(m, m), zeros(m, m), zeros(m, m), eye(m)];

    [P, ~, ~, ~, ~, ~, ~, ~] = forward(CELL, h0, c0, X, d, m, E1, E2, E3, E4);
    

    L = 0;
    for i=1:tau
        L = L - log(max(Y(:, i)' * P(:, i), 1e-8));
    end
end

function sig = sigmoid(x)
    sig = 1./(1 + exp(-x));
end

function P = softmax(s)

    P = exp(s);
    for i=1:size(s, 2)
        P(:, i) = P(:, i) ./ sum(P(:, i));
    end
end

function [grads, h, c, loss] = backpropagation(CELL, h0, c0, X, Y, d, m, tau)

    E1 = [eye(m), zeros(m, m), zeros(m, m), zeros(m, m)];
    E2 = [zeros(m, m), eye(m), zeros(m, m), zeros(m, m)];
    E3 = [zeros(m, m), zeros(m, m), eye(m), zeros(m, m)];
    E4 = [zeros(m, m), zeros(m, m), zeros(m, m), eye(m)];

    [P, H, C, A, F, I, C_tilde, O] = forward(CELL, h0, c0, X, d, m, E1, E2, E3, E4);
    grad_W = zeros(size(CELL.W));
    grad_U = zeros(size(CELL.U));
    grad_V = zeros(size(CELL.V));
    grad_b = zeros(size(CELL.b));
    grad_c = zeros(size(CELL.c));
    
    da = zeros(1, 4*m);
    dc = zeros(1, m);
    
    for t=tau:-1:1
        
        dout = -(Y(:, t)-P(:, t))';
        
        grad_V = grad_V + dout' * H(:, t+1)';
        grad_c = grad_c + dout';
        
        
        dh = da * CELL.W + dout * CELL.V;
        
        if t == tau
            dc = dh .* (O(:, t) .* (1 - tanh(C(:, t+1)).^2))';
        else
            dc = dc .* F(:, t+1)' + dh .* (O(:, t) .* (1 - tanh(C(:, t+1)).^2))';
        end
        
        do = dh .* (tanh(C(:, t+1)))';
        
        df = dc .* (C(:, t))';
        
        di = dc .* (C_tilde(:, t))';
        
        dc_tilde = dc .* (I(:, t))';
        
        da = [ df .* (sigmoid(E1*A(:, t)).*(1 - sigmoid(E1*A(:, t))))', di .* (sigmoid(E2*A(:, t)).*(1 - sigmoid(E2*A(:, t))))', dc_tilde .* (1 - tanh(E3*A(:, t)).^2)', do .* (sigmoid(E4*A(:, t)).*(1 - sigmoid(E4*A(:, t))))' ];
        
        
        grad_W = grad_W + da' * H(:, t)';
        grad_U = grad_U + da' * X(:, t)';
        grad_b = grad_b + da';
        
    end
    
    grads.W = grad_W;
    grads.U = grad_U;
    grads.V = grad_V;
    grads.b = grad_b;
    grads.c = grad_c;
    
    h = H(:, tau+1);
    c = C(:, tau+1);
    
    loss = 0;
    for i=1:tau
        loss = loss - log(Y(:, i)' * P(:, i));
    end

end

function CELLstar = AdaGrad(tweets, CELL, GDparams, ind_to_char, char_to_ind, d, m)
    
    mV = zeros(size(CELL.V));
    mW = zeros(size(CELL.W));
    mU = zeros(size(CELL.U));
    mb = zeros(size(CELL.b));
    mc = zeros(size(CELL.c));
    
    CELLstar.V = CELL.V;
    CELLstar.W = CELL.W;
    CELLstar.U = CELL.U;
    CELLstar.b = CELL.b;
    CELLstar.c = CELL.c;
    
    smooth_loss = 0;
    
    yloss = [];
    x = 0;
    
    minloss = 1e6;
    
    for i=1:GDparams.n_epochs
        
        cprev = zeros(size(CELL.W, 2), 1);
        hprev = zeros(size(CELL.W, 2), 1);
        
        for j=1:length(tweets)
            
            n = length(tweets{j});
            
            if n > 1

                X_hot = zeros(d, n);
                Y_hot = zeros(d, n);
                
                
                for s=1:n-1
                    X_hot(char_to_ind(tweets{j}(s)), s) = 1;
                    Y_hot(char_to_ind(tweets{j}(s+1)), s) = 1;
                end
                
                X_hot(char_to_ind(tweets{j}(n)), n) = 1;
                Y_hot(d, n) = 1;

                if smooth_loss == 0
                    smooth_loss = ComputeLoss(X_hot, Y_hot, CELL, hprev, cprev);
                end

                cprev = zeros(size(CELL.W, 2), 1);
                hprev = zeros(size(CELL.W, 2), 1);
                
                [grads, hprev, cprev, loss] = backpropagation(CELL, hprev, cprev, X_hot, Y_hot, d, m, n);

                mV = mV + grads.V.^2;
                mW = mW + grads.W.^2;
                mU = mU + grads.U.^2;
                mb = mb + grads.b.^2;
                mc = mc + grads.c.^2;

                CELL.V = CELL.V - (GDparams.eta./sqrt(mV + GDparams.epsilon)).*grads.V;
                CELL.W = CELL.W - (GDparams.eta./sqrt(mW + GDparams.epsilon)).*grads.W;
                CELL.U = CELL.U - (GDparams.eta./sqrt(mU + GDparams.epsilon)).*grads.U;
                CELL.b = CELL.b - (GDparams.eta./sqrt(mb + GDparams.epsilon)).*grads.b;
                CELL.c = CELL.c - (GDparams.eta./sqrt(mc + GDparams.epsilon)).*grads.c;

                smooth_loss = .999 * smooth_loss + .001 * loss;


                if mod(j+i*length(tweets), 2000) == 0
                    smooth_loss
                    yloss = [yloss, smooth_loss];
                    x = x + 1;

                    if minloss > smooth_loss
                        minloss = smooth_loss;
                        CELLstar = CELL;
                    end
                    
                    cprev = zeros(size(CELL.W, 2), 1);
                    hprev = zeros(size(CELL.W, 2), 1);
                    
                    x0 = zeros(d, 1);
                    x0(d-1) = 1;

                    synthesize(CELL, hprev, cprev, x0, d, m, ind_to_char)

                end

            end
        end
    end
    
    figure, plot(1:x, yloss), grid on

end

function num_grads = ComputeGradsNum(X, Y, CELL, h)
    for f = fieldnames(CELL)'
        disp('Computing numerical gradient for')
        disp(['Field name: ' f{1} ]);
        num_grads.(f{1}) = ComputeGradNum(X, Y, f{1}, CELL, h);
    end
end

function grad = ComputeGradNum(X, Y, f, CELL, h)
    n = numel(CELL.(f));
    grad = zeros(size(CELL.(f)));
    hprev = zeros(size(CELL.W, 2), 1);
    c0 = hprev;
    for i=1:n
        CELL_try = CELL;
        CELL_try.(f)(i) = CELL.(f)(i) - h;
        l1 = ComputeLoss(X, Y, CELL_try, hprev, c0);
        CELL_try.(f)(i) = CELL.(f)(i) + h;
        l2 = ComputeLoss(X, Y, CELL_try, hprev, c0);
        grad(i) = (l2-l1)/(2*h);
    end

end