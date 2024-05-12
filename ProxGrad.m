function [Sigma, values, para] = ProxGrad(Sigma, Lambda, iter_out, para)
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


    if isfield(para, 'EPS')
        EPS = para.EPS;
    else
        EPS = 2.2E-16;
    end


    if isfield(para, 'tol_opt')
        tol = para.tol_opt;
    else
        tol = 1e-3;
    end


    if isfield(para, 'tau')
        tau = para.tau;
    else
        tau = 1e-6;
    end


    if isfield(para, 'ObseNum')
        m = para.ObseNum;
    else
        disp('Error: No observation number, please check!!!')
    end

    if isfield(para, 'SenMatrix')
        A = para.SenMatrix;
    else
        disp('Error: No sensing matrix, please check!!!');
    end


    if isfield(para, 'ObseVec')
        Y = para.ObseVec;
    else
        disp('Error: No observation vectors, please check!!!');
    end

    phi_init = 1.1;
    phi = phi_init;

    %% Initial setup 
    for inner_iteration = 1:maxNewtonIter
        
        % init
        Sigma_inn_old = Sigma;

        % update Sigma
        while (1)
            Sigma_inn_old_inv = pinv(Sigma_inn_old);
            
            % Compute the gradient of f_1
            S = Gradient_f_1(Sigma_inn_old, para);

            % Compute the gradient of f
            g = S - tau * Sigma_inn_old_inv;

            B = Sigma_inn_old - g / phi;

            Sigma_inn = soft_threshold(B, Lambda, phi);

            if (func_g(Sigma_inn, Sigma_inn_old, Y, g, A, m, tau, phi) >= func_f(Sigma_inn, Y, A, m, tau) && (min(eig(Sigma_inn)) > 0))
                break;
            else
                phi = 2 * phi;
            end
        end

        phi = max(phi_init, phi / 2);

        Sigma = Sigma_inn;

        FSigma_pre = func_f(Sigma_inn_old, Y, A, m, tau) + sum(sum(Lambda .* abs(Sigma_inn_old)));
        FSigma = func_f(Sigma_inn, Y, A, m, tau) + sum(sum(Lambda .* abs(Sigma_inn)));


        fprintf("Iter %d: obj %f\n", inner_iteration, FSigma);


        if ((norm(Sigma_inn - Sigma_inn_old, 'fro') <= EPS) && (abs(FSigma_pre - FSigma) <= EPS))
            break;
        end

        if (inner_iteration == maxNewtonIter)
            warning("Exceed the maximum number of inner loop iterations!!!")
        end

    end

    values = func_f(Sigma, Y, A, m, tau) + sum(sum(Lambda .* abs(Sigma)));

end


% function f
function [val] = func_f(Sigma_inn, Y, A, m, tau)
    val= 0;
    for i = 1:m
        val = val + 1 / 2 * (Y(i) - A(:, i)' * Sigma_inn * A(:, i)) ^ 2;
    end
    val = val / (2 * m);
    val = val - tau * log(det(Sigma_inn));
end


% function g
function [val] = func_g(Sigma_inn, Sigma_inn_old, Y, g, A, m, tau, phi)
    val = 0;
    for i = 1:m
        val = val + 1 / 2 * (Y(i) - A(:, i)' * Sigma_inn_old * A(:, i)) ^ 2;
    end
    val = val / (2 * m);
    val = val - tau * log(det(Sigma_inn_old)) + phi / 2 * norm(Sigma_inn - Sigma_inn_old, 'fro') ^ 2 ...
        +sum(sum(g .* (Sigma_inn - Sigma_inn_old)));
end



%% soft thresholding operator
function [Z] = soft_threshold(B,Lambda,phi)

Z = sign(B).*max(abs(B)-Lambda/phi,0);

Z = Z - diag(diag(Z)) + diag(diag(B));

end