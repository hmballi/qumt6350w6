#Case Study 4
#Hilary Balli, Marco Duran Perez, George Garcia

library(mlbench)
library(skimr)
library(caret)
library(DALEX)
library(iml)
library(corrplot)
library(pamr)
library(rpart)
library(plyr)
library(C50)
library(xgboost)
library(caretEnsemble)
library(AppliedPredictiveModeling)

data(PimaIndiansDiabetes)
str(PimaIndiansDiabetes)
summary(PimaIndiansDiabetes)
?PimaIndiansDiabetes

dataPID <- na.omit(PimaIndiansDiabetes)
View(dataPID)
#Regression will be used.

#########################
#Data Splitting

#outcome variable
#diabetes2 <- c("neg", "pos")
#diabetes2.factor <- factor(diabetes2)
#diabetes <- as.numeric(diabetes2.factor)
#str(diabetes)

x <- subset(dataPID, select = -diabetes)
y <- subset(dataPID, select = diabetes)


# Plot distribution of the two-class outcome variable 'Class' to check for imbalance
barplot(table(y), 
        names.arg = c("neg", "pos"),
        col = c("blue", "green"),
        main = "Diabetes")

#skewness
skew <- lapply(x, skewness)
head(skew)

#set seed
seed <- 321 

#set training and testing data
ptrain <- createDataPartition(y$diabetes,
                              p = .75,
                              list = FALSE)
xTest <- x[-ptrain,]
yTest <- y[-ptrain,]

xTrain <- x[ptrain,]
yTrain <- y[ptrain,]

#skewed results
skimmed1 <- skim_to_wide(xTrain)
skimmed2 <- skim_to_wide(yTrain)
View(skimmed1)
View(skimmed2)

###Data Visualization

# Boxplots of predictors
boxplot(xTrain)

# Set predictors as numeric
mark <- sapply(xTrain, is.factor)
xTrainNum <- as.data.frame(lapply(xTrain, function(x) as.numeric(as.character(x))))

# Histograms of predictors and outcome variables
par(mfrow = c(3,3))
hist(xTrainNum$pregnant)
hist(xTrainNum$glucose)
hist(xTrainNum$pressure)
hist(xTrainNum$triceps)
hist(xTrainNum$insulin)
hist(xTrainNum$mass)
hist(xTrainNum$pedigree)
hist(xTrainNum$age)
hist(as.numeric(yTrain))

# Correlations
correlations <- cor(xTrainNum)
par(mfrow = c(1,1))
corrplot(correlations, method = "number", order = "hclust")

# Find and omit high correlations above threshold = 0.8
highCorr <- findCorrelation(correlations, cutoff = 0.8)
xTrainNum <- xTrainNum[, -highCorr]

###Algorithms (with standardized cleaning)

# Controlled resampling
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     summaryFunction = twoClassSummary, sampling = "smote", 
                     classProbs = TRUE, savePredictions = TRUE)

#nearest shrunken centroid
gridNSC <- data.frame(threshold = seq(0, 0.5, by = 0.01))
set.seed(seed)
modelNSC <- train(x = xTrainNum, y = yTrain, 
                  method = "pam",
                  preProc = c("center", "scale"),
                  tuneGrid = gridNSC,
                  trControl = ctrl)
modelNSC # ROC = , Sens = , Spec = 

#random forest
gridRF <- expand.grid(mtry = seq(0.5, 2, by = 0.5))
set.seed(seed)
prandomforest <- caret::train(x = xTrain[, -20], y = yTrain, 
                              method = "rf", 
                              preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "knnImpute", "zv", "nzv"), 
                              tuneGrid = gridRF,
                              trControl = ctrl)
prandomforest #accuracy = , kappa = 

#bagged trees
set.seed(seed)
ptreebag <- caret::train(x = xTrainNum, y = yTrain,
                         method = "treebag",
                         preProc = c("center", "scale", "YeoJohnson", "corr", "spatialSign", "zv", "nzv"),
                         nbagg = 50,  
                         trControl = ctrl)

ptreebag #ROC = .78, sens = .73, spec = .68

#C5.0
gridC5 <- expand.grid(trials = c(25:30), 
                      model = c("tree", "rules"),
                      winnow = c(TRUE, FALSE))
set.seed(seed)
modelC5 <- train(x = xTrainNum,
                 y = yTrain,
                 method = "C5.0",
                 preProc = c("center", "scale"),
                 tuneGrid = gridC5,
                 verbose = FALSE,
                 trControl = ctrl)
modelC5 # ROC = , Sens = , Spec = 

# eXtreme Gradient Boosted Tree
gridXGBT <- expand.grid(nrounds = c(50, 100), 
                        max_depth = c(1, 2),
                        eta = c(0.4, 0.5),
                        gamma = 0,
                        colsample_bytree = c(0.6, 0.8),
                        min_child_weight = 1,
                        subsample = c(0.5, 0.75, 1.0))
set.seed(seed)
modelXGBT <- train(x = xTrainNum, y = yTrain, 
                   method = "xgbTree",
                   tuneGrid = gridXGBT,
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelXGBT # ROC = , Sens = , Spec = 


# eXtreme Gradient Boosted DART
gridXGBD <- expand.grid(nrounds = c(50, 100), 
                        max_depth = c(1, 2),
                        eta = c(0.2, 0.3),
                        gamma = 0,
                        subsample = c(0.5, 0.75, 1.0),
                        colsample_bytree = c(0.6, 0.8),
                        rate_drop = c(0.01, 0.25),
                        skip_drop =  c(0.05, 0.25),
                        min_child_weight = 1)
set.seed(seed)
modelXGBD <- train(x = xTrainNum, y = yTrain, 
                   method = "xgbDART",
                   tuneGrid = gridXGBD,
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelXGBD # ROC = , Sens = , Spec = 


# eXtreme Gradient Boosted Linear
gridXGBL <- expand.grid(nrounds = c(40, 50, 60), 
                        lambda = 0,
                        alpha = 0,
                        eta = c(0.3, 0.4))
set.seed(seed)
modelXGBL <- train(x = xTrainNum, y = yTrain, 
                   method = "xgbLinear",
                   tuneGrid = gridXGBL,
                   preProc = c("center", "scale"), 
                   trControl = ctrl)
modelXGBL # ROC = , Sens = , Spec = 

#DALEX
set.seed(seed)
regr_rf <- train(pregnant~., data = dataPID, method = "rf", ntree = 100)

regr_gbm <- train(pregnant~. , data = dataPID, method = "gbm")

regr_nn <- train(pregnant~., data = dataPID,
                 method = "nnet",
                 linout = TRUE,
                 preProcess = c('center', 'scale'),
                 maxit = 500,
                 tuneGrid = expand.grid(size = 2, decay = 0),
                 trControl = trainControl(method = "none", seeds = 1))

### Explain Function ###

data("PimaIndiansDiabetes")

explainer_regr_rf <- DALEX::explain(regr_rf, label = "rf", 
                                    data = PimaIndiansDiabetes, y = PimaIndiansDiabetes$diabetes,
                                    colorize = FALSE)

explainer_regr_gbm <- DALEX::explain(regr_gbm, label = "gbm", 
                                     data = PimaIndiansDiabetes, y = PimaIndiansDiabetes$diabetes,
                                     colorize = FALSE)

explainer_regr_nn <- DALEX::explain(regr_nn, label = "nn", 
                                    data = PimaIndiansDiabetes, y = PimaIndiansDiabetes$diabetes,
                                    colorize = FALSE)
### DALEX Model Performance ###

mp_regr_rf <- model_performance(explainer_regr_rf)
mp_regr_gbm <- model_performance(explainer_regr_gbm)
mp_regr_nn <- model_performance(explainer_regr_nn)

mp_regr_rf
mp_regr_gbm
mp_regr_nn

plot(mp_regr_rf, mp_regr_nn, mp_regr_gbm)
plot(mp_regr_rf, mp_regr_nn, mp_regr_gbm, geom = "boxplot")

### DALEX Variable Importance ###

vi_regr_rf <- variable_importance(explainer_regr_rf, loss_function = loss_root_mean_square)
vi_regr_gbm <- variable_importance(explainer_regr_gbm, loss_function = loss_root_mean_square)
vi_regr_nn <- variable_importance(explainer_regr_nn, loss_function = loss_root_mean_square)

plot(vi_regr_rf, vi_regr_gbm, vi_regr_nn)


#IML

