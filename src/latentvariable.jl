struct VariableLatent
    user::Array{Float64,2}
    item::Array{Float64,2}
end

function svd(dataset::Persa.CFDatasetAbstract, features::Int)::VariableLatent
    matrix = Persa.getmatrix(dataset)
    (result, -) = svds(matrix, nsv = features)
    U = result.U
    V = result.Vt
    return VariableLatent(U, V')
end

Base.length(vl::VariableLatent) = maximum(size(vl))

user(vl::VariableLatent) = vl.user
item(vl::VariableLatent) = vl.item

user(vl::VariableLatent, id::Int) = user(vl)[id,:]
item(vl::VariableLatent, id::Int) = item(vl)[id,:]

Base.size(vl::VariableLatent) = (size(vl.user)[2], size(vl.item)[2])
