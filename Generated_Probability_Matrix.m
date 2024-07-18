function Sigma = Generated_Probability_Matrix(d, P, alpha)
    % d: dimension of the matrix
    % P: probability that an off-diagonal entry is set to zero
    % alpha: multiple of the identity matrix to add

    % Initialize the matrix
    Sigma = zeros(d, d);
    
    % Populate the matrix with i.i.d. entries from a uniform (-1, 1) distribution
    for i = 1:d
        for j = 1:d
            if i ~= j
                if rand() > P
                    Sigma(i, j) = -1 + 2*rand(); % Uniform distribution in (-1, 1)
                else
                    Sigma(i, j) = 0;
                end
            end
        end
    end
    
    % Add a multiple of the identity matrix
    Sigma = Sigma + alpha * eye(d);
end