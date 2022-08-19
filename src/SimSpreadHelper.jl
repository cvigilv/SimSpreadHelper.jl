module SimSpreadHelper

include("utils.jl")
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
