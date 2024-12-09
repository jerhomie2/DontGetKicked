library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)
library(stacks)
library(themis)
library(bonsai)
library(lightgbm)

train <- vroom('./training.csv', na = c("", "NA", "NULL", "NOT AVAIL"))
train <- train %>%
  slice_sample(prop = 0.01)
test <- vroom('./test.csv', na = c("", "NA", "NULL", "NOT AVAIL"))
train$IsBadBuy <- as.factor(train$IsBadBuy)
# train$MMRCurrentAuctionAveragePrice <- as.numeric(train$MMRCurrentAuctionAveragePrice)
# train$MMRCurrentAuctionCleanPrice <- as.numeric(train$MMRCurrentAuctionCleanPrice)
# train$MMRCurrentRetailAveragePrice <- as.numeric(train$MMRCurrentRetailAveragePrice)
# train$MMRCurrentRetailCleanPrice <- as.numeric(train$MMRCurrentRetailCleanPrice)
# test$MMRCurrentAuctionAveragePrice <- as.numeric(test$MMRCurrentAuctionAveragePrice)
# test$MMRCurrentAuctionCleanPrice <- as.numeric(test$MMRCurrentAuctionCleanPrice)
# test$MMRCurrentRetailAveragePrice <- as.numeric(test$MMRCurrentRetailAveragePrice)
# test$MMRCurrentRetailCleanPrice <- as.numeric(test$MMRCurrentRetailCleanPrice)


# Recipe
kick_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  step_mutate(PurchDate = as.Date(PurchDate, format = "%m/%d/%Y")) %>%
  step_date(PurchDate, features=c("dow", "month", "year")) %>%
  step_mutate(IsBadBuy = factor(IsBadBuy), skip = TRUE) %>%
  step_mutate(IsOnlineSale = factor(IsOnlineSale)) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_mutate_at(contains("PurchDate"), fn = factor) %>%
  step_mutate(Region = substr(VNZIP1, 1, 1)) %>%
  step_rm(contains('MMR')) %>%
  step_rm(BYRNO, WheelTypeID, VehYear, VNST, # noninformative
          PurchDate, #turned it into date variables
          AUCGUART, PRIMEUNIT) %>% # too many missing values
  step_novel(all_nominal_predictors()) %>% # cuz R told me to use it before step_unknown
  step_unknown(all_nominal_predictors()) %>% # calls all NA values "unknown"
  step_lencode_glm(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_other(all_nominal_predictors(), threshold=0.0001) %>%
  step_impute_mean(all_nominal_predictors()) %>%
  step_zv()
prep <- prep(kick_recipe)
bake <- bake(prep, new_data = train)

# Boosting lightgbm
boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>% #or "xgboost" but lightgbm is faster
  set_mode("classification")


boost_workflow <- workflow() %>%
  add_recipe(kick_recipe) %>%
  add_model(boost_model)

boost_tuning_grid <- grid_regular(tree_depth(),
                                  trees(),
                                  learn_rate(),
                                  levels = 5)

## Random Forest
rand_forest_mod <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees = 500) %>% 
  set_engine("ranger") %>%
  set_mode("classification")

rand_forest_wf <- workflow() %>%
  add_recipe(kick_recipe) %>%
  add_model(rand_forest_mod)

rand_forest_tuning_grid <- grid_regular(mtry(range = c(1, (ncol(bake)-1))),
                                        min_n(),
                                        levels = 5)

# BART
bart_mod <- bart(trees=500) %>% 
  set_engine("dbarts") %>% 
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(kick_recipe) %>%
  add_model(bart_mod)

bart_tuning_grid <- grid_regular(trees(),
                                 levels = 5)

# Naive Bayes
# nb_mod <- naive_Bayes(Laplace=tune(), 
#                       smoothness=tune()) %>%
#   set_mode("classification") %>%
#   set_engine("naivebayes")
# 
# nb_wf <- workflow() %>%
#   add_recipe(kick_recipe) %>%
#   add_model(nb_mod)
# 
# nb_tuning_grid <- grid_regular(Laplace(),
#                         smoothness(),
#                         levels = 5)

# stacking
folds <- vfold_cv(train, v = 5, repeats=1)
untunedModel <- control_stack_grid()

randforest_models <- rand_forest_wf %>%
  tune_grid(resamples=folds,
            grid=rand_forest_tuning_grid,
            metrics=metric_set(roc_auc),
            control=untunedModel)

lightGBM_models <- boost_workflow %>%
  tune_grid(resamples=folds,
            grid=boost_tuning_grid,
            metrics=metric_set(roc_auc),
            control=untunedModel)

bart_models <- bart_wf %>%
  tune_grid(resamples=folds,
            grid=bart_tuning_grid,
            metrics=metric_set(roc_auc),
            control=untunedModel)

# nb_models <- nb_wf %>%
#   tune_grid(resamples=folds,
#             grid=nb_tuning_grid,
#             metrics=metric_set(gain_capture),
#             control=untunedModel)


my_stack <- stacks() %>%
  add_candidates(lightGBM_models) %>%
  add_candidates(randforest_models) %>%
  add_candidates(bart_models)
  

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

## Predict
predictions <- stack_mod %>% 
  predict(new_data = test, type="prob") # "class"(yes or no) or "prob"(probability)

kaggle_submission <- predictions %>%
  mutate(RefId = test$RefId) %>%
  select(RefId, .pred_1) %>%
  rename(IsBadBuy=.pred_1)


## Write out the file
vroom_write(x=kaggle_submission, file="./stackpreds.csv", delim=",")

