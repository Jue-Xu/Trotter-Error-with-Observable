using LinearAlgebra
using DelimitedFiles

file_path2 = "matrix5.txt"
file_path = "matrix.txt"
matrix_str = readdlm(file_path, ',')
matrix_str2 = readdlm(file_path2, ',')
rows, cols = size(matrix_str)
matrix_complex = Array{Complex{Float64}}(undef, (rows, cols))
matrix_complex2 = Array{Complex{Float64}}(undef, (rows, cols))

for i in 1:rows
    for j in 1:cols
        # Preprocessing: remove parentheses and "j" symbol
        cleaned_str = replace(matrix_str[i, j], r"[()]" => "")
        matrix_complex[i, j] = parse(Complex{Float64}, cleaned_str)
    end
end

for i in 1:rows
    for j in 1:cols
        # Preprocessing: remove parentheses and "j" symbol
        cleaned_str = replace(matrix_str2[i, j], r"[()]" => "")
        matrix_complex2[i, j] = parse(Complex{Float64}, cleaned_str)
    end
end

norm = opnorm(matrix_complex, 2)
println("norm: ", norm)
println("norm: ", opnorm(matrix_complex2, 2))
# println(matrix_complex-matrix_complex2)