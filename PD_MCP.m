function [Sigma_final] = PD_MCP(Y, A, lambda, tau)

maxout = 10;
maxinn = 1e3;
eps = 1e-7;
phi_init = 1.1;


d = size(A, 2);
m = size(A, 1);

% init
I = eye(d);
Sigma_init = I;

Sigma_out = Sigma_init;


% outer loop
for q = 1:maxout
    % init
    Sigma_out_old = Sigma_out;

    % adaptive weight
    Lambda = func_MCP(lambda, Sigma_out_old);

    Sigma_inn = Sigma_out_old;
    phi = phi_init;

    % inner loop
    for t = 1:maxinn
        % init
        Sigma_inn_old = Sigma_inn;

        % update Sigma
        while 1
            Sigma_inn_old_inv = I / Sigma_inn_old;
            S = zeros(d);
            for m_t = 1:m
                S = S + A(m_t, :)' * A(m_t, :) * (Y(m_t) - A(m_t, :) * Sigma_inn_old * A(m_t, :)');
            end
            S = S ./ m;
            B = Sigma_inn_old - (-S - tau * Sigma_inn_old_inv) / phi;
            Sigma_inn = soft_threshold(B, Lambda, phi);

            if func_g(Y, A, Sigma_inn, Sigma_inn_old, Sigma_inn_old_inv, tau, m, phi) >= func_f(Y, A, Sigma_inn, tau, m)
                break;
            else
                phi = 2 * phi;
            end
        end

        phi = max(phi_init, phi/2);
        Sigma_out = Sigma_inn;

        % stopping criterion for inner loop
        if norm(Sigma_inn - Sigma_inn_old, "fro") <= eps
            break
        end
    end

    if q == 4
        break
    end

end
end



    
function [Z] = soft_threshold(B, Lambda, phi)
    Z = sign(B) .* max(abs(B) - Lambda/phi, 0);
end


% function f
function [val] = func_f(Y, A, Sigma, tau, m)
    val = 0;
    for i = 1:m
        val = val + (Y(i) - A(i, :) * Sigma * A(i, :)') ^ 2 / 2 / m;
    end
    val = val - tau * log(det(Sigma));
end


% function g
function [val] = func_g(Y, A, Sigma_inn, Sigma_inn_old, Sigma_inn_old_inv, tau, m, phi)
    val = 0;
    for i = 1:m
        val = val + (Y(i) - A(i, :) * Sigma_inn_old * A(i, :)') ^ 2 / 2 / m;
    end
    val = val - tau * log(det(Sigma_inn_old));
    S_1 = zeros(size(Sigma_inn, 1));
    for m_1 = 1:m
        S_1 = S_1 + A(m_1, :)' * A(m_1, :) * (Y(m_1) - A(m_1, :) * Sigma_inn_old * A(m_1, :)');
    end
    S_1 = S_1 ./ m;
    S_1 = S_1 - tau * Sigma_inn_old_inv;
    val = val + sum(sum((S_1) .* (Sigma_inn - Sigma_inn_old))) + phi/2*norm(Sigma_inn - Sigma_inn_old, 'fro')^2;
end


function [Lambda] = func_MCP(lambda, Sigma)
    d = size(Sigma, 1);
    % init
    Lambda = zeros(d, d);
    a = 2;

    for i = 1:d
        for j = 1:d
            if (abs(Sigma(i, j)) < a * lambda)
                Lambda(i, j) = lambda - abs(Sigma(i, j)) / a;
            end
        end
    end
end

