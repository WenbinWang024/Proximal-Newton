function Sigma = Generated_Block_Matrix(d, numGroups)
    % d: dimension of the matrix
    % numGroups: number of groups to partition the indices into
    % Ensure that d is divisible by numGroups
    if mod(d, numGroups) ~= 0
        error('The dimension d must be divisible by the number of groups.');
    end
    
    % Calculate the size of each group
    groupSize = d / numGroups;
    
    % Initialize the matrix
    Sigma = zeros(d, d);
    
    % Populate the matrix according to the given rules
    for i = 1:d
        for j = 1:d
            if i == j
                Sigma(i, j) = 1;
            elseif floor((i-1)/groupSize) == floor((j-1)/groupSize)
                Sigma(i, j) = 0.5;
            end
        end
    end
end


