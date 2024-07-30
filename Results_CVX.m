function [Sigma_pre, values_pre, para] = Results_CVX(Sigma, para)
    m = para.ObseNum;
    d = para.dim;
    A = para.SenMatrix;
    Y = para.ObseVec;
    Lambda = MCP(abs(Sigma), para);
    tau = para.tau;
    cvx_begin
        variable x(d, d) symmetric semidefinite 
        minimize (0.5 * norm(Y - diag(A' * x * A), 2) / m - tau * log_det(x) + norm(Lambda .* x, 1));
    cvx_end
    Sigma_pre = double(x);
    values_pre = cvx_optval;
end
