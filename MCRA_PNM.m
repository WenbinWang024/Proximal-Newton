function [Sigma, values, para] = MCRA_PNM(Sigma, para)
%% Outer Iterations
% INPUT:
%       Sigma: The covariance matrix
%       para: A structure containing a number of hyperparameters and loop invariants
% OUTPUT:
%       Sigma: Covariance matrix

    %% Total Out Iterations
    if isfield(para, 'inter_max_out')
        inter_max_out = para.inter_max_out;        
    else 
        inter_max_out = 8;   
    end

    if isfield(para, 'Sigma_star')
        Sigma_star = para.Sigma_star;
    else
        disp('Error: No Sigma_star!!!');
    end

    if isfield(para, 'EPS')
        EPS = para.EPS;
    else
        EPS = 2.2E-16;
    end

    values = 1e15;
    %% Adaptive Penalty for the inner ProxNewton
    for iter_out = 1:inter_max_out

        fprintf(['the outer iteration' ...
            '(the %d-th convex subproblem) is %d\n'], iter_out, iter_out);

        % STEP 1: Update penalty coefficient ${Lambda}^{(l - 1)}$
        Lambda = MCP(abs(Sigma), para);
        % Lambda = SCAD(lambda, a, Sigma);
        % Lambda = 0.5 * ones(para.dim);

        % STEP 2: Obtain ${Sigma}^{(l)}$
        valuespre = values;
        Sigmapre = Sigma;
%         if (iter_out == 1)
%             [Sigma, values, para] = ProxGrad(Sigma, Lambda, iter_out, para);
%             fprintf("Iter_out %d: number of non-zero %d\n", iter_out, nnz(Sigma));
%         else
        [Sigma, values, para] = ProxNewton(Sigma, Lambda, iter_out, para);
%         end

        if (values > valuespre)
            Sigma = Sigmapre;
            values = valuespre;
            break;
        end

        if ((abs(valuespre - values) <= EPS) && (norm(Sigmapre - Sigma, "fro") / norm(Sigma, "fro") <= EPS))
            break;
        end
   
        % STEP3: Output the difference between Sigma_star and Sigma obtained by the iter_out-th loop
        fprintf(['The Fibonacci norm of the difference in ' ...
            'the %d-th is norm(Sigma_star - Sigma, "fro") / norm(Sigma_star, "fro") = %d,\n'], ...
            iter_out, norm(Sigma_star - Sigma, "fro") / norm(Sigma_star, "fro"));

        if (iter_out == inter_max_out)
            fprintf('Loop termination\n!!!');
        end

        
    end

end
