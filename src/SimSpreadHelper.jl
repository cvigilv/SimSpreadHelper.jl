module SimSpreadHelper

using Base
using CUDA
using DelimitedFiles
using MLBase
using NamedArrays
using NetworkBasedInference
using Random
using StatsBase
using Trapz

import DelimitedFiles.writedlm
include("graphs.jl")
include("simspread.jl")

export writedlm,
    read_namedmatrix,
    k,
    cutoff,
    pcutoff,
    featurize,
    pfeaturize,
    split,
    prepare,
    prepare!,
    predict,
    predict!,
    clean!,
    save
end
