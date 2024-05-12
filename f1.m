function [f1Sigma] = f1(Sigma, para)
%% The value of the f1 function
% INPUT:
%       Sigma: The covariance matrix
%       para: A structure containing a number of hyperparameters and loop invariants
% OUTPUT:
%       f1Sigma: The value of the f1 function

    if isfield(para, 'ObseNum')
        m = para.ObseNum;
    else
        disp('Error: No observation numbers, please check!!!');
    end

    if isfield(para, 'ObseVec')
        Y = para.ObseVec;
    else
        disp('Error: No observation vectors, please check!!!');
    end

    if isfield(para, 'SenMatrix')
        A = para.SenMatrix;
    else
        disp('Error: No sensing matrix, please check!!!');
    end

    f1Sigma = 0;
    for i = 1:m
        f1Sigma = f1Sigma + 1 / 2 * (Y(i) - A(:, i)' * Sigma * A(:, i)) ^ 2;
    end
    f1Sigma = f1Sigma / (2 * m);
end