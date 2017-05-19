reload("Persa")

using Base.Test
using DecisionTree
using DatasetsCF

# write your own tests here
#@test 1 == 2
###
reload("COFILS")

dataset = DatasetsCF.MovieLens()

holdout = Persa.HoldOut(dataset, 0.9)

(ds_train, ds_test) = Persa.get(holdout)

model = COFILS.Cofils(ds_train, 10)
Persa.train!(model, ds_train)

print(Persa.aval(model, ds_test))
