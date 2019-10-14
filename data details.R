options(warn = -1)

suppressMessages(require(plyr))
suppressMessages(require(data.table))
suppressMessages(require(tidyverse))
suppressMessages(require(caret))
suppressMessages(require(highcharter))
suppressMessages(require(lubridate))


####
path.dir = getwd()
data.dir = paste0(path.dir,"/Data")




###
df = fread(paste0(data.dir,"/train_values.csv")) %>% as.data.frame()
label = fread(paste0(data.dir,"/train_labels.csv")) %>% as.data.frame()
test = fread(paste0(data.dir,"/test_values.csv")) %>% as.data.frame()

train.id = df$row_id
label = label$accepted
test.id = test$row_id

###
df =df %>% within(rm("row_id"))
test = test %>% within(rm("row_id"))
df= rbind(df,test)


df = df %>% 
  mutate(
    ffiecmedian_family_income = ifelse(is.na(ffiecmedian_family_income),-999,ffiecmedian_family_income),
    `number_of_owner-occupied_units` = ifelse(is.na(`number_of_owner-occupied_units`),-999,`number_of_owner-occupied_units`),
    co_applicant = ifelse(co_applicant==TRUE,1,0)
  )

app = aggregate(applicant_income ~ loan_amount, data = df, median)
lot.f2 = c()
for (i in df$loan_amount[is.na(df$applicant_income)]) {
  lot.f2 = c(lot.f2, which(app$loan_amount == i))
}
df$applicant_income[is.na(df$applicant_income)] = app[lot.f2,2]



df$label = c(label,rep(NA,500000))
df= df %>% 
  group_by(lender) %>% 
  mutate(lmean = mean(label, na.rm = T),
         lcont = n(),
         llmm = mean(loan_amount),
         llmin = min(loan_amount),
         llma = max(loan_amount),
         type = length(unique(loan_type)),
         app_inc = mean(applicant_income,na.rm = T),
         appm = min(applicant_income,na.rm = T),
         appma = max(applicant_income,na.rm = T)) %>% 
  ungroup() %>% 
  group_by(loan_purpose) %>% 
  mutate(pmean = mean(label,na.rm = T)) %>% 
  ungroup() %>% 
  group_by(msa_md) %>% 
  mutate(mmean = mean(label,na.rm = T)) %>% 
  ungroup()



df_train = df[1:length(train.id),]
df_test = df[(length(train.id)+1):nrow(df),]
####model 

library(xgboost)

dtrain = xgb.DMatrix(as.matrix(df_train), label=label)
dtest = xgb.DMatrix(as.matrix(df_test))


param = list(booster = "gbtree",
             objective = "binary:logistic",
             eval_metric = "error",
             eta = 0.01,
             colsample_bytree = 0.5,
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
                nfold = 5,
                print_every_n = 500,
                early_stopping_rounds = 50,
                prediction = TRUE,
                maximize = FALSE)


mod0 = xgb.train(data = dtrain,
                 params = param,
                 nrounds = 2000,
                 maximize = F,verbose = 1)

pred = predict(mod0, dtest)
a = ifelse(pred>0.5,1,0)

sub = data.frame(row_id = test.id,accepted = a)
write.csv(sub, file = "sub.csv",row.names = F)



