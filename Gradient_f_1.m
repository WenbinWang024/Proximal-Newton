function [gradient] = Gradient_f_1(Sigma, para)
%% Solve the gradient of the f1 function(Without the logdet and penalty)
% INPUT:
%       Sigma: The covariance matrix
%       para: A structure containing a number of hyperparameters and loop invariants
% OUTPUT:
%       gradient: The gradient value of f1 function

    if isfield(para, 'Q')
        Q = para.Q;
    else
        disp('Error: No Q, please check!!!')
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

    if isfield(para, 'dim')
        dim = para.dim;
    else
        dim = size(Sigma, 1);
    end

    if isfield(para, 'ObseNum')
        m = para.ObseNum;
    else
        disp('Error: No observation numbers, please check!!!');
    end

    gradient = zeros(dim);
    for i = 1:m
        gradient = gradient - Q{1, i} * (Y(i) - A(:, i)' * Sigma * A(:, i));
    end
    gradient = gradient / m;
end