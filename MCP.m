function [Lambda] = MCP(Sigma, para)
%% Computes the minimax concave penalty coefficient matrix
% INPUT:
%       Sigma: Covariance Matrix
%       para: A structure containing a number of hyperparameters and loop invariants
% OUTPUT:
%       Lambda: Coefficient Matrix

    %% Hyperparameter
        % Set the regularization parameter lambda (default 0.5）
        if isfield(para, 'lambda')
            lambda = para.lambda;
        else
            lambda = 0.5;
        end

        % Set the hyperparameters in MCP (default 3.7)
        if isfield(para, 'a_Penalty')
            a = para.a_Penalty;
        else
            a = 3.7;
        end

    Lambda = zeros(size(Sigma));
    judge = (Sigma >= 0 & Sigma <= a * lambda); 
    Lambda = Lambda + (lambda - Sigma ./ a) .* judge;

end