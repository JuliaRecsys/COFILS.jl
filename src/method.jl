abstract type Cofils <: Persa.CFModel end

mutable struct GlobalCofils <: Cofils
    slmodel::Nullable{DecisionTree.Ensemble}
    lv::LatentVariable
    features::Int
    preferences::Persa.RatingPreferences
end

mutable struct UserCofils <: Cofils
    slmodel::Nullable{DecisionTree.Ensemble}
    lv::LatentVariable
    features::Int
    means::Array{Float64}
    preferences::Persa.RatingPreferences
end

mutable struct ItemCofils <: Cofils
    slmodel::Nullable{DecisionTree.Ensemble}
    lv::LatentVariable
    features::Int
    means::Array{Float64}
    preferences::Persa.RatingPreferences
end

userlv(model::Cofils, id::Int) = user(model.lv, id)
itemlv(model::Cofils, id::Int) = item(model.lv, id)

function Cofils(dataset::Persa.CFDatasetAbstract, features::Int; normalization::Symbol = :user, extractlv::Function = svd)
    @assert in(normalization, [:user, :item, :none]) "Incorrect value of normalization. Use :user, :item or :none."

    if normalization == :user
        means = Persa.means(dataset; mode = :user, α = 10)
        return UserCofils(Nullable{DecisionTree.Ensemble}(), extractlv(dataset, features), features, means, dataset.preferences)
    elseif normalization == :item
        means = Persa.means(dataset; mode = :item, α = 10)
        return ItemCofils(Nullable{DecisionTree.Ensemble}(), extractlv(dataset, features), features, means, dataset.preferences)
    end

    return GlobalCofils(Nullable{DecisionTree.Ensemble}(), extractlv(dataset, features), features, dataset.preferences)
end

function Persa.train!(model::Cofils, dataset::Persa.CFDatasetAbstract; nfeaturestrees::Int = 10, trees::Int = 20)
    if length(model.lv) < nfeaturestrees
        warn("RandomFlorest: # features trees is greeter than the features quantity.");
        nfeaturestrees = length(model.lv);
    end

    (attributes, labels) = convert2sl(model, dataset)

    model.slmodel = Nullable{DecisionTree.Ensemble}(build_forest(labels, attributes, nfeaturestrees, trees))

    return nothing
end

function convert2sl(model::GlobalCofils, dataset::Persa.CFDatasetAbstract)
    m, n = size(model.lv)

    attributes = Array{Float64,2}(length(dataset), m + n)
    labels = Array{Float64,1}(length(dataset))

    for i=1:length(dataset)
      (u, v, r) = dataset[i]

      attributes[i, :] = convert2sl(model, u, v)
      labels[i] = r
    end

    return attributes, labels
end

function convert2sl(model::UserCofils, dataset::Persa.CFDatasetAbstract)
    m, n = size(model.lv)

    attributes = Array{Float64,2}(length(dataset), m + n)
    labels = Array{Float64,1}(length(dataset))

    for i=1:length(dataset)
      (u, v, r) = dataset[i]

      attributes[i, :] = convert2sl(model, u, v)
      labels[i] = r - model.means[u]
    end

    return attributes, labels
end

function convert2sl(model::ItemCofils, dataset::Persa.CFDatasetAbstract)
    m, n = size(model.lv)

    attributes = Array{Float64,2}(length(dataset), m + n)
    labels = Array{Float64,1}(length(dataset))

    for i=1:length(dataset)
      (u, v, r) = dataset[i]

      attributes[i, :] = convert2sl(model, u, v)
      labels[i] = r - model.means[v]
    end

    return attributes, labels
end

function convert2sl(model::Cofils, user::Int, item::Int)
    m, n = size(model.lv)

    attributes = Array{Float64,2}(1, m + n)

    attributes[1,1:m] = userlv(model, user)
    attributes[1,(m+1):(m+n)] = itemlv(model, item)

    return attributes
end

function Persa.predict(model::GlobalCofils, user::Int64, item::Int64)
    attributes = convert2sl(model, user, item)

    p = apply_forest(get(model.slmodel), attributes)[1]

    return Persa.correct(p, model.preferences)
end

function Persa.predict(model::UserCofils, user::Int64, item::Int64)
    attributes = convert2sl(model, user, item)

    p = apply_forest(get(model.slmodel), attributes)[1] + model.means[user]

    return Persa.correct(p, model.preferences)
end

function Persa.predict(model::ItemCofils, user::Int64, item::Int64)
    attributes = convert2sl(model, user, item)

    p = apply_forest(get(model.slmodel), attributes)[1] + model.means[item]

    return Persa.correct(p, model.preferences)
end

Persa.canpredict(model::Cofils, user::Int, item::Int) = true
