struct LatentVariable
    user::Array{Float64,2}
    item::Array{Float64,2}
end

function svd(dataset::Persa.CFDatasetAbstract, features::Int)::LatentVariable
    matrix = Persa.getmatrix(dataset)
    (result, -) = svds(matrix, nsv = features)
    U = result.U
    V = result.Vt
    return LatentVariable(U, V')
end

Base.length(lv::LatentVariable) = maximum(size(lv))

user(lv::LatentVariable) = lv.user
item(lv::LatentVariable) = lv.item

user(lv::LatentVariable, id::Int) = user(lv)[id,:]
item(lv::LatentVariable, id::Int) = item(lv)[id,:]

Base.size(lv::LatentVariable) = (size(lv.user)[2], size(lv.item)[2])
