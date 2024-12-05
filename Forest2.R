library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)
library(discrim)

train <- vroom('./training.csv')
test <- vroom('./test.csv')
train <- train %>% select(-AUCGUART, -PRIMEUNIT)
test <- test %>% select(-AUCGUART, -PRIMEUNIT)
train$IsBadBuy <- as.numeric(train$IsBadBuy)
train$Transmission <- as.numeric(train$Transmission)
train$Trim <- as.numeric(train$Trim)
test$Transmission <- as.numeric(test$Transmission)
test$Trim <- as.factor(test$Trim)
train$MMRCurrentAuctionAveragePrice <- as.numeric(train$MMRCurrentAuctionAveragePrice)
train$MMRCurrentAuctionCleanPrice <- as.numeric(train$MMRCurrentAuctionCleanPrice)
train$MMRCurrentRetailAveragePrice <- as.numeric(train$MMRCurrentRetailAveragePrice)
train$MMRCurrentRetailCleanPrice <- as.numeric(train$MMRCurrentRetailCleanPrice)
test$MMRCurrentAuctionAveragePrice <- as.numeric(test$MMRCurrentAuctionAveragePrice)
test$MMRCurrentAuctionCleanPrice <- as.numeric(test$MMRCurrentAuctionCleanPrice)
test$MMRCurrentRetailAveragePrice <- as.numeric(test$MMRCurrentRetailAveragePrice)
test$MMRCurrentRetailCleanPrice <- as.numeric(test$MMRCurrentRetailCleanPrice)

kick_recipe <- recipe(IsBadBuy ~ ., data = train) %>%
  step_mutate(PurchDate = as.Date(PurchDate)) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_impute_mean(MMRAcquisitionAuctionAveragePrice,
                  MMRAcquisitionRetailAveragePrice,
                  MMRAcquisitionAuctionCleanPrice,
                  MMRAcquisitonRetailCleanPrice,
                  MMRCurrentAuctionAveragePrice,
                  MMRCurrentAuctionCleanPrice,
                  MMRCurrentRetailAveragePrice,
                  MMRCurrentRetailCleanPrice) %>%
  step_impute_knn(Transmission, Trim, neighbors = 10) %>%
  
  step_zv()
prep <- prep(kick_recipe)
bake <- bake(prep, new_data = train)

my_mod <- rand_forest(mtry=31, 
                      min_n=30, 
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
vroom_write(x=kaggle_submission, file="./forestpreds2.csv", delim=",")


