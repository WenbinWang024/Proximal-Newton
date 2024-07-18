function [matrix] = Generated_Toeplitz_Matrix(dim)
    for i = 1:dim
        for j = 1:dim
            matrix(i, j) = 0.75 ^ abs(i - j);
        end
    end
end