function [Sigma, values, para] = ProxNewton(Sigma, Lambda, iter_out, para)
%% Inner Iterations
% INPUT:
%       Sigma: The covariance matrix
%       Lambda: Coefficient Matrix
%       para: A structure containing a number of hyperparameters and loop invariants
% OUTPUT:
%       Sigma: The covariance matrix

    %% Total Inner Iterations and Tolerance
    if isfield(para, 'inter_max_in')
        maxNewtonIter = para.inter_max_in;
    else
        maxNewtonIter = 1e3;
    end

    if isfield(para, 'dim')
        dim = para.dim;
    else
        dim = size(Sigma, 1);
    end

    if isfield(para, 'tol_opt')
        tol = para.tol_opt;
    else
        tol = 1e-3;
    end

    if isfield(para, 'max_lineiter')
        max_lineiter = para.max_lineiter;
    else
        max_lineiter = 1e6;
    end

    if isfield(para, 'tau')
        tau = para.tau;
    else
        tau = 1e-6;
    end

    if isfield(para, 'EPS')
        EPS = para.EPS;
    else
        EPS = 2.2E-16;
    end

    if isfield(para, 'Q')
        Q = para.Q;
    else
        disp('Error: No Q, please check!!!');
    end

    if isfield(para, 'ObseNum')
        m = para.ObseNum;
    else
        disp('Error: No observation number, please check!!!')
    end

    if isfield(para, 'Q_tol')
        Q_tol = para.Q_tol;
    else
        disp('Error: No Q_tol, please check!!!');
    end

    %% Initial setup 
    cdSweepTol = 0.0000000005;
    FSigma = 1e+15;
    FSigma1 = 1e+15;
    FSigmaprev = 1e+15;
    sigma = 0.001;

    error_occur = 0;
    l1normSigma = 0.0;
    f1Sigma = 0.0;
    logdetSigma = 0.0;
    values = 1e15;

    %% Proximal Newton Algorithm With Back-tracking Line Search
    for NewtonIter = 1:maxNewtonIter
        trgradgD = 0.0; % maintain trgradG during coordiante descent
        normD = 0.0;
        diffD = 0.0;
        subgrad = 1e+15;

        %% Compute the Newton Direction
        if (NewtonIter == 1 && isdiag(Sigma))
  
            % Initial D
            D = zeros(dim);
            [FSigma, Sigma, trgradgD, D] = DiagNewton(Sigma, Lambda, D, trgradgD, para);
        else
            % Initial D
            D = zeros(dim);

            numActive = 0;
            subgrad = 0;

            % Store free set
            Set_free = [];

            % Construct Free Set
            % Stopping tolerance for CG when computing gradient
            g_tol = 1e-10;

            % Compute W
%             W = zeros(dim);
%             Sigma_symmetric_copy = Make_X_Symmetric_Copy(Sigma);
%             for i = 1:dim
%                 ei = zeros(dim, 1);
%                 ei(i) = 1;
%                 [W(i, :)] = ComputeAinvb(Sigma_symmetric_copy, ei, W(i, :), dim, g_tol);
%             end
%             for i = 1:dim
%                 for j = 1:(i - 1)
%                     W(j, i) = W(i, j);
%                 end
%             end
            W = pinv(Sigma);


            % Compute the gradient of f_1
            S = Gradient_f_1(Sigma, para);
            
            % Compute the gradient of f
            g = S - tau * W;

            % Identify free sets
            for j = 1:dim
                for k = 1:j
                    if (Sigma(j, k) ~= 0 || abs(g(j, k)) > Lambda(j, k))
                        numActive = numActive + 1;
                        Set_free = [Set_free; j, k];
                        if (Sigma(j, k) > 0)
                            g(j, k) = g(j, k) + Lambda(j, k);
                        elseif (Sigma(j, k) < 0)
                            g(j, k) = g(j, k) - Lambda(j, k);
                        else
                            g(j, k) = abs(g(j, k)) - Lambda(j, k);
                        end
                        subgrad = subgrad + abs(g(j, k));
                    end
                end
            end

            % coordinate descent
            for cdSweep = 1:(1 + mod(NewtonIter, 50)  / 3)
                diffD = 0; 
                
                % Coordinate descent updates
                for num = 1:size(Set_free, 1)
                    j = Set_free(num, 1);
                    k = Set_free(num, 2);
                    Djk = D(j, k);
                    a = tau * W(j, k) ^ 2 + Q_tol(j, k);
                    if (j ~= k)
                        a = a + tau * W(j, j) * W(k, k);
                    end
                    b = S(j, k) - tau * W(j, k) + tau * W(:, j)' * D * W(:, k);
                    for i = 1:m
                        b = b + Q{1, i}(:, j)' * D * Q{1, i}(:, k) / m;
                    end

                    l = Lambda(j, k) / a;
                    c = Sigma(j, k) + D(j ,k);                
                    f = b / a;
                    mu = 0;
                    normD = normD - abs(D(j, k));
                    if (c > f)
                        mu = -f - l;
                        if (c + mu < 0)
                            mu = -c;
                            D(j, k) = -Sigma(j, k);
                            if (j ~= k)
                                D(k, j) = D(j, k);
                            end
                        else
                            D(j, k) = D(j, k) + mu;
                            if (j ~= k)
                                D(k, j) = D(j, k);
                            end
                        end
                    else
                        mu = -f + l;
                        if (c + mu > 0)
                            mu = -c;
                            D(j, k) = -Sigma(j, k);
                            if (j ~= k)
                                D(k, j) = D(j, k);
                            end
                        else
                            D(j, k) = D(j, k) + mu;
                            if (j ~= k)
                                D(k, j) = D(j, k);
                            end
                        end
                    end
                    diffD = diffD + abs(mu);
                    normD = normD + abs(D(j, k));
                    if (j == k)
                        trgradgD = trgradgD + (S(j, k) - tau * W(j, k)) * (D(j, k) - Djk);
                    else
                        trgradgD = trgradgD + (S(j, k) - tau * W(j, k)) * (D(j, k) - Djk) * 2;
                    end
                end

                if (diffD <= normD * cdSweepTol)
                    break;
                end

                if (diffD > 1e10)
                    error_occur = 1;
                    break;
                end
            end
        end

        if (error_occur == 1)
            break;
        end

        %% Line Search
        alpha = 1.0;
        l1normSigmaD = 0.0;
        FSigma1prev = 1e+15;
        for lineiter = 1:max_lineiter
            Sigma_alphaD = Sigma + alpha * D;

            l1normSigma1 = sum(sum(abs(Sigma_alphaD) .* Lambda));
            f1Sigma1 = f1(Sigma_alphaD, para);

            logdetSigma1 = log(det(Sigma_alphaD));
            flag_pd = (min(eig(Sigma_alphaD)) > 0);
%             Sigma_alphaD_copy = Make_X_Copy(Sigma_alphaD);
%             [logdetSigma1, flag_pd] = ispos_computelogdet(Sigma_alphaD_copy, 1e-5);

            
            if (~flag_pd)
                fprintf('Line search step size %f. Lack of positive definiteness \n', alpha);
                alpha = alpha * 0.5;
                continue;
            end
            

            FSigma1 = (f1Sigma1 + l1normSigma1) - tau * logdetSigma1;

            if (alpha == 1.0)
                l1normSigmaD = l1normSigma1;
            end
            if (FSigma1 <= FSigma + alpha * sigma * (trgradgD + l1normSigmaD - l1normSigma) || normD == 0)
                FSigmaprev = FSigma;
                FSigma = FSigma1;
                l1normSigma = l1normSigma1;
                logdetSigma = logdetSigma1;
                f1Sigma = f1Sigma1;
                Sigma = Sigma_alphaD;
                break;
            end
            if (FSigma1prev < FSigma1)
                FSigmaprev = FSigma;
                l1normSigma = l1normSigma1;
                logdetSigma = logdetSigma1;
                f1Sigma = f1Sigma1;
                Sigma = Sigma_alphaD;
                break;
            end
            FSigma1prev = FSigma1;
            alpha = alpha * 0.5;
        end

        fprintf("Iter %d: obj %f\n", NewtonIter, FSigma1);

        values = FSigma1;

        para.process{iter_out, NewtonIter} = norm(Sigma - para.Sigma_star, "fro");

        % Chech for convergence
        if (subgrad * alpha >= l1normSigma * tol && (abs((FSigma - FSigmaprev) / FSigma) >= EPS))
            continue;
        end
        break;
    end

end





