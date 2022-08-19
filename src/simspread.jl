using Base
using CUDA
using Random
using StatsBase
using NamedArrays
using NetworkBasedInference

"""
    cutoff(x::T, α::T, β::T, weighted::Bool=false) where {T<:Float64}

wl-SimSpread similarity cutoff function

# Arguments
- `x::T` : Value to apply criteria
- `α::T` : Strong-ties threshold
- `β::T` : Weak-ties threshold
- `weighted::Bool` : Apply weighting function to outcome (default = False)
"""
function cutoff(x::T, α::T, β::T, weighted::Bool=false) where {T<:Float64}
    @assert α ≥ β "β can't be greater than α!"
    if x == 0
        return 0
    end    # Check if edge exist (val ≥ 0)
    w = weighted ? x : 1        # Define weighting scheme
    x ≥ α ? w : x < β ? w : 0 # Filter edge by cutoffs α & β
end

"""
    pcutoff(x::T, α::T, β::T, weighted::Bool=false) where {T<:Float64}

wl-SimSpread similarity probabilistic cutoff function

# Arguments
- `x::T` : Value to apply criteria
- `α::T` : Strong-ties threshold
- `β::T` : Weak-ties probability
- `weighted::Bool` : Apply weighting function to outcome (default = False)
"""
function pcutoff(x::T, α::T, β::T, weighted::Bool=false) where {T<:Float64}
    if x == 0
        return 0
    end # Check if edge exist (val ≥ 0)
    p = rand()
    w = weighted ? x : 1        # Define weighting scheme
    x ≥ α ? w : p ≤ β ? w : 0 # Filter edge by cutoffs α & β
end

"""
    prepare!(DT::T, DF::T, Cs::AbstractVector) where {T<:NamedMatrix}

Prepare compound-feature-drug-target network adjacency matrix for *de novo* NBI prediction.

# Arguments
- `DT::NamedMatrix`: Drug-Target adjacency matrix
- `DF::NamedMatrix`: Drug-Feature adjacency matrix
- `Cs::AbstractVector`: Compounds to predict targets

# Extended help
This implementation is for k-fold or leave-one-out cross-validation.
"""
function prepare!(DT::T, DF::T, Cs::AbstractVector) where {T<:NamedMatrix}
    @assert size(DT, 1) == size(DF, 1) "Different number of compounds!"

    # Get names from matrices
    Fs = [f for f in names(DF, 2) if f ∉ Cs]
    Ds = [d for d in names(DF, 1) if d ∉ Cs]
    Ts = names(DT, 2)

    if Fs[1] == Ds[1]
        Fs = ["f$f" for f in Fs]
        setnames!(DF, Fs, 2)
    end

    println(Fs[begin:5])
    println(Ds)
    println(Ts[begin:5])
    println(Cs)

    # Get dimensions of network
    Nc = length(Cs)
    Nf = length(Fs)
    Nd = length(Ds)
    Nt = length(Ts)

    # Construct trilayered graph adjacency matrix
    Mcc = zeros(Nc, Nc)
    Mcd = zeros(Nc, Nd)
    Mcf = DF[Cs, Fs].array
    Mct = zeros(Nc, Nt)

    Mdc = Mcd'
    Mdd = zeros(Nd, Nd)
    Mdf = DF[Ds, Fs].array
    Mdt = DT[Ds, Ts].array

    Mfc = Mcf'
    Mfd = Mdf'
    Mff = zeros(Nf, Nf)
    Mft = zeros(Nf, Nt)

    Mtc = Mct'
    Mtd = Mdt'
    Mtf = Mft'
    Mtt = zeros(Nt, Nt)

    A = Matrix(
        [ Mcc Mcd Mcf Mct;
          Mdc Mdd Mdf Mdt;
          Mfc Mfd Mff Mft;
          Mtc Mtd Mtf Mtt]
    )

    namedA = NamedArray(A, (vcat(Cs, Ds, Fs, Ts), vcat(Cs, Ds, Fs, Ts)))
    namedB = deepcopy(namedA)
    namedB[Cs,:] .= 0
    namedB[:,Cs] .= 0

    return namedA, namedB
end

"""
    prepare(dts::T, dfs::T) where {T<:Tuple{NamedMatrix,NamedMatrix}}

Prepare compound-feature-drug-target network adjacency matrix for *de novo* NBI predictions.

# Arguments
- `dts::Tuple{NamedMatrix,NamedMatrix}` : Drug-Target adjacency matrices
- `dfs::Tuple{NamedMatrix,NamedMatrix}` : Drug-Feature adjacency matrices

# Extended help
This implementation is for time-split cross-validation.
"""
function prepare(dts::T, dfs::T) where {T<:Tuple{NamedMatrix,NamedMatrix}}
    # Unpack matrices tuples
    DT₀, DT₁ = dts
    DF₀, DF₁ = dfs

    @assert size(DT₀, 2) == size(DT₁, 2) "Different number of targets!"
    @assert size(DF₀, 2) == size(DF₁, 2) "Different number of features!"

    # Get names from matrices
    F₀ = names(DF₀, 2)
    D₀ = names(DT₀, 1)
    T₀ = names(DT₀, 2)
    D₁ = names(DT₁, 1)

    if F₀ == D₀
        F₀ = ["_$f" for f in F₀]
    end

    # Get dimensions of network
    Nd = length(D₀)
    Nc = length(D₁)
    Nt = length(T₀)
    Nf = length(F₀)

    # Construct trilayered graph adjacency matrix
    Mcc = zeros(Nc, Nc)
    Mcd = zeros(Nc, Nd)
    Mcf = DF₁.array
    Mct = zeros(Nc, Nt)
    Mdc = Mcd'
    Mdd = zeros(Nd, Nd)
    Mdf = DF₀.array
    Mdt = DT₀.array
    Mfc = Mcf'
    Mfd = Mdf'
    Mff = zeros(Nf, Nf)
    Mft = zeros(Nf, Nt)
    Mtc = Mct'
    Mtd = Mdt'
    Mtf = Mft'
    Mtt = zeros(Nt, Nt)

    A = Matrix(
        [ Mcc Mcd Mcf Mct;
          Mdc Mdd Mdf Mdt;
          Mfc Mfd Mff Mft;
          Mtc Mtd Mtf Mtt]
    )

    namedA = NamedArray(A, (vcat(D₁, D₀, F₀, T₀), vcat(D₁, D₀, F₀, T₀)))
    namedB = deepcopy(namedA)
    namedB[D₁,:] .= 0
    namedB[:,D₁] .= 0

    return namedA, namedB
end

"""
    split(DT::NamedArray, k::Int64, rng::Int64)

Split all possible `D` into `k` groups for cross-validation.

# Long description
Split drugs `D` into `k` groups, extract their edges and append to cross-validation group.

# Arguments
- `DT::AbstractMatrix`: Drug-Target rectangular adjacency matrix.
- `k::Int64`: Number of groups to use in data splitting.
- `rng::Int64`: Seed used for data splitting.

"""
function Base.split(G::NamedArray, ngroups::Int64; seed::Int64 = 1)
    # Get array of drugs in adjacency matrix
    D = names(G,1)

    # Assign fold to edges of graph
    shuffle!(MersenneTwister(seed), D)
    groups = [ [] for _ in 1:ngroups ]

    for (i, dᵢ) in enumerate(D)
        foldᵢ = mod(i, ngroups) + 1
        push!(groups[foldᵢ], dᵢ)
    end

    return groups
end


"""
    clamp(val::Number, vmin::Number, vmax::Number)

Restrict a value to a given range.

# Arguments
- `val::Number` : Value to clamp
- `vmin::Number` : Value floor
- `vmax::Number` : Value roof
"""
function clamp(val::Number, vmin::Number, vmax::Number)
    val < vmin ? vmin : val > vmax ? vmax : val
end

"""
    featurize(M::NamedArray, α::Float64, β::Float64, weighted::Bool)

Convert continuous feature into binary feature based in 2 cutoffs: (i) α for strong-ties and
(ii) β for weak-ties. Weighted version of function weights binary features with it's real
value.

# Arguments
- `M::AbtractMatrix`: Continuous feature matrix
- `α::Float64`: Strong-ties cutoff
- `β::Float64`: Weak-ties cutoff
- `weighted::Bool`: Flag for feature weighting using real value
"""
function featurize(M::NamedArray, α::Float64, β::Float64, weighted::Bool)
    # Filter matrix
    Mf = copy(M)
    Mf.array = cutoff.(M.array, α, β, weighted)
    setnames!(Mf, ["f$f" for f in names(Mf, 2)], 2)
    return Mf
end

"""
    pfeaturize(M::AbtractMatrix, α::Float64, β::Float64, weighted::Bool)

Convert continuous feature into binary feature based in 2 parameters: (i) α for strong-ties
cutoff and (ii) β for weak-ties probability. Weighted version of function weights binary 
features with it's real value.

# Arguments
- `M::AbtractMatrix`: Continuous feature matrix
- `α::Float64`: Strong-ties cutoff
- `β::Float64`: Weak-ties probability
- `weighted::Bool`: Flag for feature weighting using real value
"""
function pfeaturize(M::NamedArray, α::Float64, β::Float64, weighted::Bool)
    # Filter matrix
    Mf = copy(M)
    Mf.array = pcutoff.(M.array, α, β, weighted)
    setnames!(Mf, ["f$f" for f in names(Mf, 2)])
    return Mf
end

"""
    zfeaturize(M::AbtractMatrix, α::Float64, β::Float64, weighted::Bool)

Convert continuous feature into binary feature based in 2 parameters: (i) α for strong-ties
z-cutoff and (ii) β for weak-ties z-cutoff. Weighted version of function weights binary
features with it's real value.

# Arguments
- `M::AbtractMatrix`: Continuous feature matrix
- `α::Float64`: Strong-ties z-cutoff
- `β::Float64`: Weak-ties z-cutoff
- `weighted::Bool`: Flag for feature weighting using real value
"""
function zfeaturize(M::NamedArray, zα::Float64, zβ::Float64, weighted::Bool)
    # Convert z-cutoff to cutoff
    μ = mean(vec(M))
    σ = std(vec(M))
    α, β = map(z -> clamp(z * σ + μ, 0.0, 1.0), [zα, zβ])

    # Filter matrix
    Mf = copy(M)
    Mf.array = cutoff.(M.array, α, β, weighted)
    setnames!(Mf, ["f$f" for f in names(Mf, 2)])
    return Mf
end


"""
    predict(A::NamedMatrix, B::NamedMatrix, names::Tuple; GPU::Bool)

TODO: Add short description to `predict`

# Arguments
- `A::NamedMatrix`: Compound-Feature-Drug-Target initial resources adjacency matrix
- `B::NamedMatrix`: Feature-Drug-Target initial resources adjacency matrix
- `names::Tuple`: Rows & columns named indices
- `GPU::Bool`: (default = false)
"""
function predict(A::T, B::T, names::Tuple; GPU::Bool = false) where {T<:NamedMatrix}
    # GPU calculations helper functions
    _useGPU(x::AbstractArray) = GPU ? CuArray(x) : x

    # Target prediction using NBI
    W = denovoNBI(B.array)
    F = begin
        F = copy(A)
        F.array = _useGPU(A.array) * _useGPU(W)^2
        return F
    end

    R = F[Symbol.(names[1]), Symbol.(names[2])]
    return R
end

"""
    predict(I::Tuple{T,T}, DT::T; GPU::Bool) where {T<:NamedArray}

TODO: Add short description to `predict`

# Arguments
- `I::Tuple{NamedMatrix,NamedMatrix}`: *De novo* initial resources adjacency matrices
- `DT::NamedMatrix`: Drug-target adjacency matrix
- `GPU::Bool`: Use GPU acceleration for calculation (default = false)
"""
function predict(I::Tuple{T,T}, DT::NamedMatrix; GPU::Bool = false) where {T<:NamedMatrix}
    # GPU calculations helper functions
    _useGPU(x::AbstractArray) = GPU ? CuArray(x) : x

    # Target prediction using NBI
    A, B = I
    W = denovoNBI(B.array)
    F = begin
        F = copy(A)
        F.array = _useGPU(A.array) * _useGPU(W)^2
        return F
    end

    R = F[names(DT, 1), names(DT, 2)]
    return R
end

"""
    clean!(R::NamedArray, A::NamedArray)

Flag errors from cross-validation splitting in place.

# Arguments
- `R::NamedArray`: Drug-Target predictions matrix
- `A::NamedArray`: *de novo* NBI initial resources adjacency matrix
"""
function clean!(R::T, A::T) where {T<:NamedArray}
    # Clean predictions adjacancy matrix R from disconnected targets
    for (tᵢ, k) in zip(names(A,1), k(A))
        if k == 0
            R[:,tᵢ] .= -99
            R[tᵢ,:] .= -99
        end
    end
end

"""
    save(filepath::String, R::NamedMatrix, DT::NamedMatrix)

Store predictions as a table in the given file path.

# Arguments
- `filepath::String`: Output file path
- `R::NamedArray`: Drug-target predictions matrix
- `DT::NamedMatrix`: Drug-target interactions adjacency matrix

# Extended help
Table format is:
```
fold, compound ID, target ID, score, TP
```
"""
function save(filepath::String, R::NamedMatrix, DT::NamedMatrix)
    # Get name arrays
    Cnames = names(DT, 1)
    Tnames = names(DT, 2)

    # Save file
    open(filepath, "a+") do f
        for Cᵢ in Cnames, Tᵢ in Tnames
            write(f, "$(findfirst(id -> id == Cᵢ, Cnames)), $Cᵢ, $Tᵢ, $(R[Cᵢ,Tᵢ]), $(DT[Cᵢ,Tᵢ])\n")
        end
    end
end

"""
    save(filepath::String, fold::Int64, R::NamedMatrix, DT::NamedMatrix)

Store predictions as a table in the given file path.

# Arguments
- `filepath::String`: Output file path
- `fold::Int64`: Fold ID
- `R::NamedArray`: Drug-target predictions matrix
- `DT::NamedMatrix`: Drug-target interactions adjacency matrix

# Extended help
Table format is:
```
fold, compound ID, target ID, score, TP
```
"""
function save(filepath::String, fold::Int64, R::NamedMatrix, DT::NamedMatrix)
    # Get name arrays
    Cnames = names(DT, 1)
    Tnames = names(DT, 2)

    # Save file
    open(filepath, "a+") do f
        for Cᵢ in Cnames, Tᵢ in Tnames
            write(f, "$fold, $Cᵢ, $Tᵢ, $(R[Cᵢ,Tᵢ]), $(DT[Cᵢ,Tᵢ])\n")
        end
    end
end
