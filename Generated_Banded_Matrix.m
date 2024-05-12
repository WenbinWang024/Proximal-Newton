function [Sigma] = Generated_Banded_Matrix(dim, difference)
    Sigma = zeros(dim);
    for i = 1: dim
        for j = 1:dim
            if (abs(i - j) <= difference)
                Sigma(i, j) = 1 - abs(i - j) / difference;
            end
        end
    end
end