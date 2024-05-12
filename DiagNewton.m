function [FSigma, Sigma, trgradgD, D] = DiagNewton(Sigma, Lambda, D, trgradgD, para)
%% Treat with Diagonal matrix using Newton method
% INPUT:
%       Sigma: The covariance matrix
%       Lambda: Coefficient Matrix
%       D: the Newton direction(the initial value is zero matrix)
%       trgradgD:
%       para:
% OUTPUT:
%       FSigma:
%       Sigma:
%       trgradgD:
%       D: the updated Newton direction

    %%


    %% Update the diagonal matrix
    total = 0;  % for maintaining total amount of nonzero D
    logdet = 0;
    l1normSigma = 0;
    f1Sigma = 0;

    % Q_tol, tau
    if isfield(para, 'Q_tol')
        Q_tol = para.Q_tol;
    else
        disp('Error: No Q_tol, please check!!!');
    end

    if isfield(para, 'tau')
        tau = para.tau;
    else
        tau = 1e-6;
    end

    if isfield(para, 'dim')
        dim = para.dim;
    else
        dim = size(Sigma, 1);
    end

    W = eye(dim) / Sigma;
    S = Gradient_f_1(Sigma, para);  % the gradient value of f1 function
    
    % Computing updates
    %j >= k
    for j = 1:dim
        for k = 1:j
            % k < j
            if (j ~= k)
                b = S(j, k);
                a = tau * W(j, j) * W(k, k)  + Q_tol(j ,k);
                l = Lambda(j, k) / a;
                f = b / a;
                mu = 0;
                if (0 > f)
                    mu = -f - l;
                    if (mu > 0)
                        D(j, k) = mu;
                    end
                else
                    mu = -f + l;
                    if (mu < 0)
                        D(j, k) = mu;
                    end
                end
                if (D(j, k) ~= 0)
                    D(k, j) = D(j, k);
                    total = total + 1;
                    trgradgD = trgradgD + D(j, k) * b * 2;
                end
            % j == k    
            else
                c = Sigma(j, k);
                logdet = logdet + log(c);
                l1normSigma = l1normSigma + abs(c) * Lambda(j, k);
                a = tau * W(j, k) ^ 2 + Q_tol(j, k);
                b = S(j, k) - tau * W(j, k);                
                l = Lambda(j, k) / a;
                f = b / a;
                mu = 0;
                if (c > f)
                    mu = -f - l;
                    if (c + mu < 0)
                        D(j, k) = -c;
                    else
                        D(j, k) = mu;
                    end
                else
                    mu = -f + l;
                    if (c + mu > 0)
                        D(j ,k) = -c;
                    else
                        D(j, k) = mu;
                    end
                end
                if (D(j, k) ~= 0)
                    total = total + 1;
                    trgradgD = trgradgD + D(j, k) * b;
                end
            end
        end
    end

    f1Sigma = f1(Sigma, para);
    FSigma = f1Sigma - tau * logdet + l1normSigma;

end