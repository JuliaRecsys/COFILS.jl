reload("Persa")
reload("COFILS")
using Base.Test
using DecisionTree
using DatasetsCF

# write your own tests here
#@test 1 == 2
###
dataset = DatasetsCF.MovieLens()

holdout = Persa.HoldOut(dataset, 0.9)

(ds_train, ds_test) = Persa.get(holdout)

features = 5

matrix = Persa.getMatrix(ds_train)

(result, -) = svds(matrix, nsv = features)
U = result.U
V = result.Vt

training_set = Array{Float64,2}(length(ds_train), 2 .* features)
labels = Array{Float64,1}(length(ds_train))

for i=1:length(ds_train)
  (u, v, r) = ds_train[i]

  training_set[i,1:features] = U[u,:]
  training_set[i,(features+1):end] = V[v,:]
  labels[i] = r
end


model = build_forest(labels, training_set, 2, 20)

result = apply_forest(model, training_set)
