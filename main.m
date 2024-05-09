clear,clc
% rng(0);

dim = 100;
Sigma_star = Generated_Banded_Matrix(100, 10);
% full(sprandsym(50, 0.032, 0.2, 1));

error_para = 1e-1;
error_list_non = [];
error_list_L1 = [];
m_list = [80, 120, 180, 200, 240, 280, 320, 360, 400, 440, 500, 550, 600];
for i = 1:13
    m = m_list(i);
    para.a_Penalty = 2;
    para.inter_max_out = 1e3;
    para.inter_max_in = 1e5;
    para.tol_opt = 1e-6;
    para.max_lineiter = 100;
    para.EPS = 2.22E-16;
    para.Sigma_star = Sigma_star;
    
    subvar = size(Sigma_star, 2);
    A = randn(subvar, m); 
    
    Y = [];
    for i = 1:m
        Y = [Y; A(:, i)' * Sigma_star * A(:, i) + error_para * (2 * rand() - 1)];
    end
    % + error_para * (2 * rand() - 1)
    
    Sigma = eye(subvar);
    
    Q = cell(1, m);
    for i = 1:m
        Q{1, i} = A(:, i) * A(:, i)';
    end
    Q_tol = zeros(subvar);
    for j = 1:subvar
        for k = 1:j
            if (j ~= k)
                for i = 1:m
                    Q_tol(j, k) = Q_tol(j, k) + Q{1, i}(j, k) * Q{1, i}(j, k) + Q{1, i}(j, j) * Q{1, i}(k, k);
                end
                Q_tol(k, j) = Q_tol(j, k);
            else
                for i = 1:m
                    Q_tol(j, k) = Q_tol(j, k) + Q{1, i}(j, j) * Q{1, i}(j, j);
                end
            end
        end
    end
    para.Q = Q;
    para.Q_tol = Q_tol;
    para.SenMatrix = A;
    para.ObseVec = Y;
    para.dim = subvar;
    para.ObseNum = m;
    para.lambda = 0.1;    
    para.tau = 1e-6;
    para.process = cell(1000, 100000);
    
    [Sigma_non, values_non, para] = MCRA_PNM(Sigma, para);
    non_zero_count = nnz(Sigma_star);
    error_list_non = [error_list_non; non_zero_count, norm(Sigma_non - Sigma_star, 'fro'), norm(Sigma_star, 'fro')];
    [Sigma_L1, values_L1, para] = MCRA_PNM_L1(Sigma, para);
    error_list_L1 = [error_list_L1; non_zero_count, norm(Sigma_L1 - Sigma_star, 'fro'), norm(Sigma_star, 'fro')];
end

% save("data.mat")
