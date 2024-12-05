library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)

train <- vroom('./training.csv')
#train <- train %>%
#  slice_sample(prop = 0.05)
test <- vroom('./test.csv')
train <- train %>% select(-AUCGUART, -PRIMEUNIT)
test <- test %>% select(-AUCGUART, -PRIMEUNIT)
train$IsBadBuy <- as.factor(train$IsBadBuy)
train$MMRCurrentAuctionAveragePrice <- as.numeric(train$MMRCurrentAuctionAveragePrice)
train$MMRCurrentAuctionCleanPrice <- as.numeric(train$MMRCurrentAuctionCleanPrice)
train$MMRCurrentRetailAveragePrice <- as.numeric(train$MMRCurrentRetailAveragePrice)
train$MMRCurrentRetailCleanPrice <- as.numeric(train$MMRCurrentRetailCleanPrice)
test$MMRCurrentAuctionAveragePrice <- as.numeric(test$MMRCurrentAuctionAveragePrice)
test$MMRCurrentAuctionCleanPrice <- as.numeric(test$MMRCurrentAuctionCleanPrice)
test$MMRCurrentRetailAveragePrice <- as.numeric(test$MMRCurrentRetailAveragePrice)
test$MMRCurrentRetailCleanPrice <- as.numeric(test$MMRCurrentRetailCleanPrice)

kick_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  step_mutate(PurchDate = as.Date(PurchDate, format = "%m/%d/%Y")) %>%
  step_date(PurchDate, features=c("dow", "month", "doy", "year")) %>%
  step_rm(PurchDate) %>%
  step_mutate_at(contains("PurchDate"), fn = factor) %>%
  step_mutate(Region = substr(VNZIP1, 1, 1)) %>%
  step_interact(terms = ~ WarrantyCost:VehicleAge + IsOnlineSale:VehicleAge) %>%
  step_regex(Trim, pattern = "([a-zA-Z]+)", result = "TrimType") %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(IsBadBuy)) %>%
  step_impute_knn(Transmission,
                  MMRAcquisitionAuctionAveragePrice,
                  MMRAcquisitionRetailAveragePrice,
                  MMRAcquisitionAuctionCleanPrice,
                  MMRAcquisitonRetailCleanPrice,
                  MMRCurrentAuctionAveragePrice,
                  MMRCurrentAuctionCleanPrice,
                  MMRCurrentRetailAveragePrice,
                  MMRCurrentRetailCleanPrice,
                  Trim,
                  impute_with = imp_vars(Auction,
                                         VehYear,
                                         VehicleAge,
                                         Make,
                                         Model,
                                         SubModel,
                                         Color,
                                         WheelTypeID,
                                         WheelType,
                                         VehOdo,
                                         Nationality,
                                         Size,
                                         TopThreeAmericanName,
                                         BYRNO,
                                         VNZIP1,
                                         VNST,
                                         VehBCost,
                                         IsOnlineSale,
                                         WarrantyCost),
                  neighbors = 10) %>%
  step_mutate(AcquisitionPriceRatio = MMRAcquisitionAuctionAveragePrice / VehBCost,
              RetailPriceRatio = MMRCurrentRetailAveragePrice / VehBCost) %>%
  step_normalize(all_numeric(), -all_outcomes()) %>%
  step_zv() %>%
  step_corr(all_numeric(), threshold = 0.9)
prep <- prep(kick_recipe)
bake <- bake(prep, new_data = train)

my_mod <- rand_forest(mtry = tune(), 
                      min_n = tune(), 
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(kick_recipe) %>%
  add_model(my_mod)

# Grid of values to tune over
tuning_grid <- grid_regular(mtry(range = c(1,30)),
                            min_n(),
                            levels = 3)

## Split data for CV
folds <- vfold_cv(train, v = 3, repeats=1)

## Run the CV
CV_results <- forest_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(accuracy)) #, f_meas, sens, recall, spec, precision, accuracy)) #Or leave metrics NULL

## Find Best Tuning Parameters
bestTune <- CV_results %>%
  select_best(metric = "accuracy")

## Finalize the Workflow & fit it
final_wf <- forest_wf %>%
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
vroom_write(x=kaggle_submission, file="./forestpreds3.csv", delim=",")


