library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)

train <- vroom('./training.csv')
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
  step_date(PurchDate, features=c("dow", "month", "decimal", "doy", "year")) %>%
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
                  impute_with = imp_vars(PurchDate,
                                         Auction,
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
                  neighbors = 25) %>%
  step_zv()
prep <- prep(kick_recipe)
bake <- bake(prep, new_data = train)

my_mod <- rand_forest(mtry = 1, 
                      min_n = 40, 
                      trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

forest_wf <- workflow() %>%
  add_recipe(kick_recipe) %>%
  add_model(my_mod) %>%
  fit(data = train)

## Predict
predictions <- forest_wf %>% 
  predict(new_data = test, type="class") # "class"(yes or no) or "prob"(probability)

kaggle_submission <- predictions %>%
  mutate(RefId = test$RefId) %>%
  select(RefId, .pred_class) %>%
  rename(IsBadBuy=.pred_class)


## Write out the file
vroom_write(x=kaggle_submission, file="./forestpreds4.csv", delim=",")


