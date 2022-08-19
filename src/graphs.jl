"""
    k(G::AbstractMatrix)

Get node degrees from adjacency matrix

# Arguments
- `M::AbstractMatrix` : Matrix to parse
"""
k(vᵢ::Integer, G::AbstractMatrix) = count(!iszero, G[vᵢ, :])
k(eᵢ::AbstractVector) = count(!iszero, eᵢ)
k(G::AbstractMatrix) = mapslices(k, G; dims=2)
