module SimSpreadHelper

include("src/utils.jl")
include("src/graphs.jl")
include("src/simspread.jl")

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
