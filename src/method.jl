mutable struct Cofils <: Persa.CFModel
    slmodel::Nullable{DecisionTree.Ensemble}
    vl::VariableLatent
    features::Int
    usersmean::Array{Float64}
    preferences::Persa.RatingPreferences
end

uservl(model::Cofils, id::Int) = user(model.vl, id)
itemvl(model::Cofils, id::Int) = item(model.vl, id)

function Cofils(dataset::Persa.CFDatasetAbstract, features::Int, extractvl::Function = svd)
    usersmean = zeros(dataset.users)
    return Cofils(Nullable{DecisionTree.Ensemble}(), extractvl(dataset, features), features, Persa.shrunkUserMean(dataset, 10), dataset.preferences)
end

function Persa.train!(model::Cofils, dataset::Persa.CFDatasetAbstract; nfeaturestrees::Int = 10, trees::Int = 20)
    if length(model.vl) < nfeaturestrees
        warn("RandomFlorest: # features trees is greeter than the features quantity.");
        nfeaturestrees = length(model.vl);
    end

    (attributes, labels) = convert2sl(model, dataset)

    model.slmodel = Nullable{DecisionTree.Ensemble}(build_forest(labels, attributes, nfeaturestrees, trees))

    return nothing
end

function convert2sl(model::Cofils, dataset::Persa.CFDatasetAbstract)
    m, n = size(model.vl)

    attributes = Array{Float64,2}(length(dataset), m + n)
    labels = Array{Float64,1}(length(dataset))

    for i=1:length(dataset)
      (u, v, r) = dataset[i]

      attributes[i,1:m] = uservl(model, u)
      attributes[i,(m+1):(m+n)] = itemvl(model, v)
      labels[i] = r - model.usersmean[u]
    end

    return attributes, labels
end

function convert2sl(model::Cofils, user::Int, item::Int)
    m, n = size(model.vl)

    attributes = Array{Float64,2}(1, m + n)

    attributes[1,1:m] = uservl(model, user)
    attributes[1,(m+1):(m+n)] = itemvl(model, item)

    return attributes
end

function Persa.predict(model::Cofils, user::Int64, item::Int64)
    attributes = convert2sl(model, user, item)

    p = apply_forest(get(model.slmodel), attributes)[1] + model.usersmean[user]

    return Persa.correct(p, model.preferences)
end

Persa.canpredict(model::Cofils, user::Int, item::Int) = true
