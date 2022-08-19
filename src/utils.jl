using NamedArrays
using DelimitedFiles
import DelimitedFiles.writedlm

"""
    _namedmatrix2matrix(M::NamedMatrix)

Convert NamedMatrix to Base.Array with row / column of names
"""
_namedmatrix2matrix(M::NamedMatrix) = vcat(["" names(M, 2)...], hcat(names(M, 1), M))

writedlm(io::IO, x::NamedMatrix{T} where {T}) = writedlm(io, _namedmatrix2matrix(x))
writedlm(io::AbstractString, x::NamedMatrix{T} where {T}) = writedlm(io, _namedmatrix2matrix(x))
writedlm(io::IO, x::NamedMatrix{T} where {T}, delimiter::Char) = writedlm(io, _namedmatrix2matrix(x), delimiter)
writedlm(io::AbstractString, x::NamedMatrix{T} where {T}, delimiter::Char) = writedlm(io, _namedmatrix2matrix(x), delimiter)

"""
    _parse_matrix(M::AbstractMatrix, rows::Bool, cols::Bool; type::Type)

Convert matrix with row/column names to NamedMatrix (assumes names are in firt row/column).

# Arguments
- `M::AbstractMatrix` : Matrix to parse
- `rows::Bool` : Matrix has row names (default = false)
- `cols::Bool` : Matrix has column names (default = false)
- `type::Type` : Type of matrix values (default = Any)
"""
function _parse_matrix(M::AbstractMatrix, rows::Bool=false, cols::Bool=false; type::Type=Any)
    # Extract values from matrix
    c_idx = rows ? 2 : 1
    r_idx = cols ? 2 : 1
    values = parse.(type, M[r_idx:end, c_idx:end])

    # Extract dimensions names
    row_names = rows ? [i for i in String.(M[r_idx:end, 1])] : ["R#$i" for i in 1:size(values, 1)]
    col_names = cols ? [i for i in String.(M[1, c_idx:end])] : ["C#$i" for i in 1:size(values, 2)]

    namedM = NamedArray(values, (row_names, col_names))
    namedM = namedM[sort(row_names), sort(col_names)]

    return namedM
end

"""
    read_namedmatrix(filepath::String, valuetype::Type = FLoat64filepath::String, valuetype::Type)

Load a matrix with named indices as a NamedArray.

# Arguments
- `filepath::String` : File path of matrix to load
- `delimiter::Char` : Delimiter character between values in matrix (default = ' ')
- `valuetype::Type` : Type of values contained in matrix (default = Float64)
"""
function read_namedmatrix(filepath::String, delimiter::Char=' ', valuetype::Type=Float64)
    M = readdlm(filepath, delimiter, String)
    return _parse_matrix(M, true, true; type=valuetype)
end

function read_namedmatrix(filepath::String, delimiter::Char=' ', valuetype::Type=Float64; rows::Bool = true, cols::Bool = true)
    M = readdlm(filepath, delimiter, String)
    return _parse_matrix(M, rows, cols; type=valuetype)
end
