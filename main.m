clear,clc
rng(0);

dim = 50;
Sigma_star = Generated_Toeplitz_Matrix(dim);
% Generated_Banded_Matrix(50, 5);
% full(sprandsym(50, 0.032, 0.2, 1));

error_para = 1e-1;
error_list_non = [];
error_list_non_norm2 = [];
error_list_L1 = [];
error_list_L1_norm2 = [];
error_list_hybrid = [];
FPR_non = [];
FPR_L1 = [];
TPR_non = [];
TPR_L1 = [];
m_list = [100, 500, 1000, 1500];
for idx_list = 1:4
    m = m_list(idx_list);
    para.a_Penalty = 2;
    para.inter_max_out = 10;
    para.inter_max_in = 1e5;
    para.tol_opt = 1e-6;
    para.max_lineiter = 1e6;
    para.EPS = 2.22E-6;
    para.Sigma_star = Sigma_star;
    
    subvar = size(Sigma_star, 2);
    A = randn(subvar, m); 

    num_samples = 50;
    mu = zeros(1, dim);
    samples = mvnrnd(mu, Sigma_star, num_samples);
    Sigma_sample = cov(samples);
    
    Y = [];
    for i = 1:m
        Y = [Y; A(:, i)' * Sigma_sample * A(:, i) + error_para * randn()];
    end
    % + error_para * (2 * rand() - 1)
    

    % cross validation
    folds = 5; % 交叉验证折数
    bestTau = 0;
    bestlambda = 0;
    minError = inf;

    % The possible value of the hyperparameter
    tauValues = logspace(-8, -4, 5);
    lambdaValues = 0.1:0.1:0.5;

    % 对每一对超参数进行交叉验证
    for tau = tauValues
        for lambda = lambdaValues
            % 初始化交叉验证误差
            cvError = 0;

            % 进行交叉验证
            for fold = 1:folds
                % 划分训练集和数据集
                validationIndices = (m / folds) * (fold - 1) + 1 : (m / folds) * fold;
                trainingIndices = setdiff(1:m, validationIndices);
                yTrain = Y(trainingIndices);
                aTrain = A(:, trainingIndices);
                yValidation = Y(validationIndices);
                aValidation = A(:, validationIndices);
                para.SenMatrix = aTrain;
                para.ObseVec = yTrain;
                para.lambda = lambda;    
                para.tau = tau;
                para.dim = subvar;
                para.ObseNum = m * (folds - 1) / folds;
                para.process = cell(1000, 10000);
                numTrain = para.ObseNum;
                
                Sigma = eye(subvar);
                Q = cell(1, numTrain);
                for i = 1:numTrain
                    Q{1, i} = A(:, i) * A(:, i)';
                end
                Q_tol = zeros(subvar);
                for j = 1:subvar
                    for k = 1:j
                        if (j ~= k)
                            for i = 1:numTrain
                                Q_tol(j, k) = Q_tol(j, k) + Q{1, i}(j, k) * Q{1, i}(j, k) + Q{1, i}(j, j) * Q{1, i}(k, k);
                            end
                            Q_tol(k, j) = Q_tol(j, k);
                        else
                            for i = 1:numTrain
                                Q_tol(j, k) = Q_tol(j, k) + Q{1, i}(j, j) * Q{1, i}(j, j);
                            end
                        end
                    end
                end
                Q_tol = Q_tol / numTrain;

                para.Q = Q;
                para.Q_tol = Q_tol;

                % 训练模型并计算验证误差
                [Sigma_pre, values_pre, para] = MCRA_PNM(Sigma, para); % trainModel
                % compute error
                Y_pre = [];
                for idx = 1:(m / folds)
                    Y_pre = [Y_pre; aValidation(:, idx)' * Sigma_pre * aValidation(:, idx)];
                end
                cvError = cvError + norm(Y_pre - yValidation, 'fro');
            end
            
            % 更新最佳超参数和最小误差
            if cvError < minError
                minError = cvError;
                bestTau = tau;
                bestlambda = lambda;
            end
        end
    end

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
    Q_tol = Q_tol / m;

    para.Q = Q;
    para.Q_tol = Q_tol;
    para.SenMatrix = A;
    para.ObseVec = Y;
    para.dim = subvar;
    para.ObseNum = m;
    para.lambda = bestlambda;   
    para.tau = bestTau;
    para.process = cell(1000, 10000);


    [Sigma_non, values_non, para] = MCRA_PNM(Sigma, para);
    non_zero_count = nnz(Sigma_star);
    error_list_non = [error_list_non; non_zero_count, norm(Sigma_non - Sigma_star, 'fro'), norm(Sigma_star, 'fro'), norm(Sigma_non - Sigma_star, 'fro') / norm(Sigma_star, 'fro')];
    error_list_non_norm2 = [error_list_non_norm2; non_zero_count, norm(Sigma_non - Sigma_star), norm(Sigma_star), norm(Sigma_non - Sigma_star) / norm(Sigma_star)];
    FPR_num = 0;
    for idx1 = 1:dim
        for idx2 = 1:dim
            if Sigma_non(idx1, idx2) ~= 0 && Sigma_star(idx1, idx2) == 0
                FPR_num = FPR_num + 1;
            end
        end
    end
    FPR_non = [FPR_non; non_zero_count, FPR_num / dim * dim - non_zero_count];
    TPR_num = 0;
    for idx1 = 1:dim
        for idx2 = 1:dim
            if Sigma_non(idx1, idx2) ~= 0 && Sigma_star(idx1, idx2) ~= 0
                TPR_num = TPR_num + 1;
            end
        end
    end
    TPR_non = [TPR_non; non_zero_count, TPR_num / non_zero_count];
    [Sigma_L1, values_L1, para] = MCRA_PNM_L1(Sigma, para);
    error_list_L1 = [error_list_L1; non_zero_count, norm(Sigma_L1 - Sigma_star, 'fro'), norm(Sigma_star, 'fro'), norm(Sigma_L1 - Sigma_star, 'fro') / norm(Sigma_star, 'fro')];
    error_list_L1_norm2 = [error_list_L1_norm2; non_zero_count, norm(Sigma_L1 - Sigma_star), norm(Sigma_star), norm(Sigma_L1 - Sigma_star) / norm(Sigma_star)];
    FPR_num = 0;
    for idx1 = 1:dim
        for idx2 = 1:dim
            if Sigma_L1(idx1, idx2) ~= 0 && Sigma_star(idx1, idx2) == 0
                FPR_num = FPR_num + 1;
            end
        end
    end
    FPR_L1 = [FPR_L1; non_zero_count, FPR_num / dim * dim - non_zero_count];
    TPR_num = 0;
    for idx1 = 1:dim
        for idx2 = 1:dim
            if Sigma_L1(idx1, idx2) ~= 0 && Sigma_star(idx1, idx2) ~= 0
                TPR_num = TPR_num + 1;
            end
        end
    end
    TPR_L1 = [TPR_L1; non_zero_count, TPR_num / non_zero_count];
    % [Sigma_hybrid, values_hybrid, para] = MCRA_PNM_Hybrid(Sigma, para);
    % error_list_hybrid = [error_list_hybrid; non_zero_count, norm(Sigma_hybrid - Sigma_star, 'fro'), norm(Sigma_star, 'fro'), norm(Sigma_hybrid - Sigma_star, 'fro') / norm(Sigma_star, 'fro')];
end

save("data.mat")
