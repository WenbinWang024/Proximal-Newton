clear,clc
error = [];
error_1 = [];
mu = [80 120 180 200 240 280 320 360 400 440 500 550 600];
for num=1:10
    for k = 1:13
        x = zeros(50);
        m = mu(k);
        d = 50;
        A = randn(m, d);
        Sigma_star = full(sprandsym(d, 0.098, 0.2, 1));
        for i = 1:m
            Y(i) = A(i, :) * Sigma_star * A(i, :)' + 1e-4 * (rand() - 1);
        end
        Y = Y';
        lam = 0.5;
        disp('L1 Penalty!!!')
        cvx_begin
            variable x(50, 50) semidefinite
            minimize (0.5 * norm(Y - diag(A * x * A'), 2) + lam * norm(x, 1));
        cvx_end
        error_1 = norm(x - Sigma_star, "fro") / norm(Sigma_star, "fro");
    end
end



disp('Non-convex Penalty!!!')
for iter = 1:10
    Lambda = MCP(x, lam);
    
end
error_2 = norm(x - Sigma_star, "fro") / norm(Sigma_star, "fro")




function [Lambda] = MCP(Sigma, lambda)
    a = 2;
    Lambda = zeros(size(Sigma));
    judge = (Sigma >= 0 & Sigma <= a * lambda); 
    Lambda = Lambda + (lambda - Sigma ./ a) .* judge;
end