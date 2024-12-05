library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)

train <- vroom('./training.csv')
test <- vroom('./test.csv')
train$IsBadBuy <- as.factor(train$IsBadBuy)
train$MMRCurrentAuctionAveragePrice <- as.numeric(train$MMRCurrentAuctionAveragePrice)
train$MMRCurrentAuctionCleanPrice <- as.numeric(train$MMRCurrentAuctionCleanPrice)
train$MMRCurrentRetailAveragePrice <- as.numeric(train$MMRCurrentRetailAveragePrice)
train$MMRCurrentRetailCleanPrice <- as.numeric(train$MMRCurrentRetailCleanPrice)

head(train)
colSums(is.na(train))
nrow(train)

kick_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  # step_naomit(all_predictors()) %>% # change later
  # step_mutate_at(all_nominal_predictors(), fn = as.factor) %>% # change later
  step_mutate(PurchDate = mdy(PurchDate)) %>%
  step_date(PurchDate, features="dow") %>%
  step_mutate(PurchDate_dow = as.factor(PurchDate_dow)) %>%
  step_date(PurchDate, features="month") %>%
  step_mutate(PurchDate_month = as.factor(PurchDate_month)) %>%
  step_date(PurchDate, features="year") %>%
  step_mutate(PurchDate_year = as.factor(PurchDate_year)) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_knn(Transmission,
                  impute_with = imp_vars(-Trim,
                                         -MMRAcquisitionAuctionAveragePrice,
                                         -MMRAcquisitionRetailAveragePrice,
                                         -MMRAcquisitionAuctionCleanPrice,
                                         -MMRAcquisitonRetailCleanPrice)) %>%
  step_impute_knn(MMRAcquisitionAuctionAveragePrice,
                  MMRAcquisitionRetailAveragePrice,
                  MMRAcquisitionAuctionCleanPrice,
                  MMRAcquisitonRetailCleanPrice,
                  impute_with = imp_vars(-Trim)) %>%
  step_impute_knn(Trim) %>%
  step_zv()
prep <- prep(kick_recipe)
bake <- bake(prep, new_data = train)

my_mod <- naive_Bayes(Laplace=tune(), 
                      smoothness=tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(kick_recipe) %>%
  add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- nb_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc,accuracy)) #, f_meas, sens, recall, spec, precision, accuracy)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "roc_auc")

## Finalize the Workflow & fit it
final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
predictions <- final_wf %>%
  predict(new_data = test, type="class") # "class"(yes or no) or "prob"(probability)

kaggle_submission <- predictions %>%
  mutate(RefId = test$RefId) %>%
  select(RefId, .pred_class) %>%
  rename(IsBadBuy=.pred_class)


## Write out the file
vroom_write(x=kaggle_submission, file="./nbpreds.csv", delim=",")
  
