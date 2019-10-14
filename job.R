
library(xgboost)

dtrain = xgb.DMatrix(as.matrix(df_train), label=label2)
dtest = xgb.DMatrix(as.matrix(df_test))


param = list(booster = "gbtree",
             objective = "binary:logistic",
             eval_metric = "error",
             eta = 0.01,
             colsample_bytree = 0.7,
             max_depth = 3,
             min_child_weight = 0,
             nthread = 4,
             gamma = 0,
             subsample = 0.8
)

watchlist= list(train = dtrain)

fit_cv = xgb.cv(params = param,
                data = dtrain,
                watchlist = watchlist,
                nrounds = 10000,
                nfold = 5,
                print_every_n = 500,
                early_stopping_rounds = 50,
                prediction = TRUE,
                maximize = FALSE)



