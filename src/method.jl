type Cofils <: Persa.CFModel
  slmodel::DecisionTree.Ensemble
  userfeatures::Array{Float64,2}
  itemfeatures::Array{Float64,2}
  features::Int
  usersmean::Array{Float64}
  preferences::Persa.RatingPreferences
end

function Cofils(dataset::Persa.CFDatasetAbstract, features::Int)
  matrix = Persa.getMatrix(dataset)
  (result, -) = svds(matrix, nsv = features)
  U = result.U
  V = result.Vt

  usersmean = zeros(dataset.users)

  return Cofils(DecisionTree.Ensemble([]), U, V, features, Persa.shrunkUserMean(dataset, 10), dataset.preferences)
end

function Persa.train!(model::Cofils, dataset::Persa.CFDatasetAbstract; nfeaturestrees::Int = 10, trees::Int = 20)
  if (model.features .* 2) < nfeaturestrees
    warn("RandomFlorest: # features trees is greeter than the features quantity.");
    nfeaturestrees = model.features;
  end

  training_set = Array{Float64,2}(length(dataset), 2 .* model.features)
  labels = Array{Float64,1}(length(dataset))

  for i=1:length(dataset)
    (u, v, r) = dataset[i]

    training_set[i,1:model.features] = model.userfeatures[u,:]
    training_set[i,(model.features+1):end] = model.itemfeatures[v,:]
    labels[i] = r - model.usersmean[u]
  end

  model.slmodel = build_forest(labels, training_set, nfeaturestrees, trees)

  return nothing
end

function Persa.predict(model::Cofils, user::Int64, item::Int64)
  return Persa.correct(apply_forest(model.slmodel, vcat(model.userfeatures[user,:], model.itemfeatures[item,:])')[1] + model.usersmean[user], model.preferences)
end

Persa.canPredict(model::Cofils, user::Int, item::Int) = true
