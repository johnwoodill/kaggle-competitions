library(randomForest)
library(tidyverse)
library(glmnet)
library(caret)
library(ncvreg)
library(kernlab)
library(gbm)
library(nnet)
library(xgboost)
library(MASS)

source("func.R")

#--------------------------
# Modeling Strategy
#--------------------------

dat <- readRDS("data/red_clean.Rds")



dat <- readRDS("data/red_clean_three_groups.Rds")

filename <- "figures/cv_model_runs_all_var_three_groups_white"

dat$quality <- as.factor(dat$quality)
# dat <- dplyr::select(dat, quality, alcohol, volatile_acidity, citric_acid, density, pH, sulphates)

#  #Randomly shuffle the data
dat <- dat[sample(nrow(dat)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)

# pcadat <- readRDS("data/pca_red.Rds")
# pcadat$quality <- factor(pcadat$quality)

# Logistic Regression
lreg <- data.frame()
lreg2 <- data.frame()
for (j in 1:100){
  #  #Randomly shuffle the data
  dat <- dat[sample(nrow(dat)),]
  
  #Create 10 equally size folds
  folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)

  for (i in 1:10){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    train <- dat[-testIndexes, ]
    test <- dat[testIndexes, ]
    
    # PCA
    pca <- prcomp(dplyr::select(dat, -quality), scale = TRUE, center = TRUE)
    pca_train <- as.data.frame(predict(pca, dplyr::select(train, -quality))[, 1:8])
    train <- data.frame(pca_train, quality = train$quality)
    
    pca_test <- as.data.frame(predict(pca, dplyr::select(test, -quality))[, 1:8])
    test <- data.frame(pca_test, quality = test$quality)
        
    results = cv_multinom(train, test)
    bind <- data.frame(acc = results$accuracy, kappa = results$kappa)
    lreg = rbind(lreg, bind)
  }
  bind <- data.frame(acc = mean(lreg$acc), kappa = mean(lreg$kappa))
  lreg2 = rbind(lreg2, bind)
  lreg = data.frame()
}
lreg2
mean(lreg2$acc)
mean(lreg2$kappa)


# Random Forest
rfdat = data.frame()
rfdat2 = data.frame()
for (j in 1:100){
  #  #Randomly shuffle the data
  dat <- dat[sample(nrow(dat)),]
  
  #Create 10 equally size folds
  folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)

  for (i in 1:10){
      testIndexes <- which(folds==i,arr.ind=TRUE)
      test <- dat[testIndexes, ]
      train <- dat[-testIndexes, ]
      results = cv_rf(train, test)
      bind <- data.frame(acc = results$accuracy, kappa = results$kappa)
      rfdat = rbind(rfdat, bind)
  }
    bind <- data.frame(acc = mean(rfdat$acc), kappa = mean(rfdat$kappa))
    rfdat2 = rbind(rfdat2, bind)
    rfdat = data.frame()
    print(j)
}
mean(rfdat2$acc)
mean(rfdat2$kappa)

# Boosted Random Forest
# brfdat = data.frame()
# brfdat2 = data.frame()
# for (j in 1:100){
#   #  #Randomly shuffle the data
#   dat <- dat[sample(nrow(dat)),]
#   
#   #Create 10 equally size folds
#   folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)
# 
#   for (i in 1:10){
#       testIndexes <- which(folds==i,arr.ind=TRUE)
#       test <- dat[testIndexes, ]
#       train <- dat[-testIndexes, ]
#       results = cv_brf(train, test)
#       bind <- data.frame(acc = results$accuracy, kappa = results$kappa)
#       brfdat = rbind(brfdat, bind)
#   }
#     bind <- data.frame(acc = mean(brfdat$acc), kappa = mean(brfdat$kappa))
#     brfdat2 = rbind(brfdat2, bind)
#     brfdat = data.frame()
#     print(j)
# }
# mean(brfdat2$acc)
# mean(brfdat2$kappa)
# 
# 4 (depth)
#        acc     kappa
# 1 0.5986124 0.3334635
# 
# 5
#         acc     kappa
# 1 0.6049057 0.3449436


# rfdat <- data.frame()
# for (ntree in seq(100, 5000, 100)){
#   for (mtry in 1:30){
#       for (i in 1:10){
#         testIndexes <- which(folds==i,arr.ind=TRUE)
#         test <- red[testIndexes, ]
#         train <- red[-testIndexes, ]
#         results = cv_rf_tune(train, test, ntree = ntree, mtry = mtry)
#         bind <- data.frame(ntree = ntree, mtry = mtry, acc = results$accuracy, kappa = results$kappa)
#         rfdat = rbind(rfdat, bind)
#         print(ntree)
# }}}
#
# for (i in 1:10){
#     testIndexes <- which(folds==i,arr.ind=TRUE)
#     test <- red[testIndexes, ]
#     train <- red[-testIndexes, ]
#     results = cv_rf(train, test)
#     bind <- data.frame(acc = results$accuracy, kappa = results$kappa)
#     rfdat = rbind(rfdat, bind)
# }

# rfdat
# rfdat %>% group_by(ntree, mtry) %>% summarise(acc = mean(acc),
#                                               kappa = mean(kappa)) %>% 
# arrange(-kappa)
# mean(rfdat$acc)
# mean(rfdat$kappa)

# Tuning

# best.iter <- gbm.perf(gbmmod, method = "cv")
# print(best.iter)

# # Gradient boosting
# gbmdat <- data.frame()  
# gbmdat2 <- data.frame()
# for (j in 1:100){
#   #  #Randomly shuffle the data
#   dat <- dat[sample(nrow(dat)),]
#   
#   #Create 10 equally size folds
#   folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)
#   
#   for (i in 1:10){
#     testIndexes <- which(folds==i,arr.ind=TRUE)
#     test <- dat[testIndexes, ]
#     train <- dat[-testIndexes, ]
#     results = cv_gbm(train, test)
#     bind <- data.frame(acc = results$accuracy, kappa = results$kappa)
#     gbmdat = rbind(gbmdat, bind)
#   }
#     bind <- data.frame(acc = mean(gbmdat$acc), kappa = mean(gbmdat$kappa))
#     gbmdat2 = rbind(gbmdat2, bind)
#     gbmdat = data.frame()
# }
# mean(gbmdat2$acc)
# mean(gbmdat2$kappa)


parameters <- list(
  # General Parameters
  booster            = "gbtree",      
  silent             = 0,           

    # Booster Parameters
  # eta                = 0.03,    # learning rate          
  # gamma              = 0.7,     # Minimum loss reduction required to make a further partition on a leaf node of the tree.            
  # max_depth          = 8,       # Maximum depth of a tree         
  # min_child_weight   = 2,       # Minimum sum of instance weight (hessian) needed in a child.     
  # subsample          = .9,      # Subsample ratio of the training instances. (default = 1)           
  # colsample_bytree   = .5,      # Subsample ratio of columns when constructing each tree          
  # colsample_bylevel  = 1,       # Subsample ratio of columns for each split, in each level   
  # lambda             = 1,       # L2 regularization term on weights
  # alpha              = 0,       # L1 regularization term on weights

  # Task Parameters
  objective          = "multi:softmax",   # default = "reg:linear", multiclasses
  # eval_metric        = "merror",          # Evaluation metrics for validation data, a default metric will be assigned according to objective
  num_class          = length(levels(dat$quality)) + 1,                 # number of classes
  seed               = 1               # reproducability seed
              # , tree_method = "hist"
              # , tree_method = "hist"
              # , grow_policy = "lossguide"  # Controls a way new nodes are added to the tree. (depthwise: split at nodes closest to the root.
                                           # lossguide: split at nodes with highest loss change.)
)




# XGradient boosting
xgbmdat <- data.frame()  
xgbmdat2 <- data.frame()

for (j in 1:100){
  #  #Randomly shuffle the data
  dat <- dat[sample(nrow(dat)),]
  
  #Create 10 equally size folds
  folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)
    
  for (i in 1:10){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test <- dat[testIndexes, ]
    train <- dat[-testIndexes, ]
    results = cv_xgbm(train, test)
    bind <- data.frame(acc = results$accuracy, kappa = results$kappa)
    xgbmdat = rbind(xgbmdat, bind)
  }
    bind <- data.frame(acc = mean(xgbmdat$acc), kappa = mean(xgbmdat$kappa))
    xgbmdat2 = rbind(xgbmdat2, bind)
    xgbmdat = data.frame()
}
mean(xgbmdat2$acc)
mean(xgbmdat2$kappa)


# SVM
# svmdat <- data.frame()  
# svmdat2 <- data.frame()
# for (j in 1:100){
#   #  #Randomly shuffle the data
#   dat <- dat[sample(nrow(dat)),]
#   
#   #Create 10 equally size folds
#   folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)
#   for (i in 1:10){
#     testIndexes <- which(folds==i,arr.ind=TRUE)
#     test <- dat[testIndexes, ]
#     train <- dat[-testIndexes, ]
#     results = cv_ksvm(train, test)
#     bind <- data.frame(acc = results$accuracy, kappa = results$kappa)
#     svmdat = rbind(svmdat, bind)
#   }
#     bind <- data.frame(acc = mean(svmdat$acc), kappa = mean(svmdat$kappa))
#     svmdat2 = rbind(svmdat2, bind)
#     svmdat = data.frame()
# }
# mean(svmdat2$acc)
# mean(svmdat2$kappa)

# Linear Discriminant Analysis
ldadat <- data.frame()  
ldadat2 <- data.frame()
for (j in 1:100){
  #  #Randomly shuffle the data
  dat <- dat[sample(nrow(dat)),]
  
  #Create 10 equally size folds
  folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)

  for (i in 1:10){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    test <- dat[testIndexes, ]
    train <- dat[-testIndexes, ]
    
    pca <- prcomp(dplyr::select(dat, -quality), scale = TRUE, center = TRUE)
    pca_train <- as.data.frame(predict(pca, dplyr::select(train, -quality))[, 1:8])
    train <- data.frame(pca_train, quality = train$quality)
    
    pca_test <- as.data.frame(predict(pca, dplyr::select(test, -quality))[, 1:8])
    test <- data.frame(pca_test, quality = test$quality)
    
    
    results = cv_lda(train, test)
    bind <- data.frame(acc = results$accuracy, kappa = results$kappa)
    ldadat = rbind(ldadat, bind)
  }
    bind <- data.frame(acc = mean(ldadat$acc), kappa = mean(ldadat$kappa))
    ldadat2 = rbind(ldadat2, bind)
    ldadat = data.frame()
}

mean(ldadat2$acc)
mean(ldadat2$kappa)


pdat <- data.frame(Model = c("Logistic", "Random Forest", "Extreme-GB", "LDA"),
                   Value = c(mean(lreg2$acc), mean(rfdat2$acc),
                             mean(xgbmdat2$acc), mean(ldadat2$acc), mean(lreg2$kappa),
                             mean(rfdat2$kappa), mean(xgbmdat2$kappa), 
                             mean(ldadat2$kappa)),
                   SE    =   c(sd(lreg2$acc), sd(rfdat2$acc),
                             sd(xgbmdat2$acc), sd(ldadat2$acc), sd(lreg2$kappa),
                             sd(rfdat2$kappa), sd(xgbmdat2$kappa), 
                             sd(ldadat2$kappa)),
                   Type = rep(c("Accuracy", "Kappa", "SD Accuracy", "SD Kappa"), each = 4))


ord <- pdat %>% 
  dplyr::filter(Type == "Accuracy") %>% 
  arrange(-Value)

ggplot(filter(pdat, Type %in% c("Accuracy", "Kappa")), aes(x=Model, y=Value, group = Type, fill = Type)) + 
  theme_tufte(11) +
  geom_bar(stat = "identity", position = "dodge") + 
  geom_errorbar(aes(ymin = Value - SE*1.96, ymax = Value + SE*1.96), 
                width=.2,
        position=position_dodge(.9)) +
  scale_x_discrete(limits = c(as.character(ord$Model))) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") +
  geom_text(aes(y = -0.02, label=round(Value, 3)), position=position_dodge(width=0.9)) +
  theme(legend.title = element_blank()) +
  scale_y_continuous(breaks = seq(0, 1, .1), limits = c(-0.02,1)) +
  ylab("Accuracy/Kappa") +
  xlab(NULL) 
ggsave(paste0(filename, ".pdf"), width = 6, height = 4)

saveRDS(pdat, paste0(filename, ".Rds"))
pdat <- readRDS(paste0(filename, ".Rds"))



