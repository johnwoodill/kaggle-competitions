library(randomForest)
library(caret)

source("func.R")

dat <- readRDS("data/red_clean.Rds")
dat$quality <- as.factor(dat$quality)
# dat <- dplyr::select(dat, quality, alcohol, volatile_acidity, citric_acid, density, pH, sulphates)

#  #Randomly shuffle the data
dat <- dat[sample(nrow(dat)),]

#Create 10 equally size folds
folds <- cut(seq(1,nrow(dat)),breaks=10,labels=FALSE)

 rfdat <- data.frame()
 rfdat2 <- data.frame()
for (ntree in seq(100, 2000, 100)){
   for (mtry in 1:8){
       for (i in 1:10){
         testIndexes <- which(folds==i,arr.ind=TRUE)
         test <- dat[testIndexes, ]
         train <- dat[-testIndexes, ]
         results = cv_rf_tune(train, test, ntree = ntree, mtry = mtry)
         bind <- data.frame(ntree = ntree, mtry = mtry, acc = results$acc, kappa = results$kappa)
         rfdat = rbind(rfdat, bind)
         
       }
     bind <- data.frame(ntree = ntree, mtry = mtry, acc = mean(rfdat$acc), kappa = mean(rfdat$kappa))
     rfdat2 = rbind(rfdat2, bind)
     rfdat <- data.frame()
     print(ntree)
     }
    }

 saveRDS(rfdat2, "data/rf_tuning.Rds")
 rfdat2 <- readRDS("data/rf_tuning.Rds")
 
#  > head(arrange(rfdat2, -kappa))
#   ntree mtry       acc     kappa
# 1  1000    2 0.7163876 0.5402479
# 2   400    3 0.7157508 0.5392008
# 3   900    1 0.7163640 0.5381396
# 4   700    3 0.7138640 0.5358308
# 5   400    6 0.7119733 0.5347292
# 6   300    2 0.7132586 0.5342499

 ntree_dat <- rfdat2 %>% 
   filter(mtry == 2) %>% 
   group_by(ntree) %>% 
   summarise(acc= mean(acc),
             kappa = mean(kappa)) %>% 
   ungroup()
 
 mtry_dat <- rfdat2 %>% 
   group_by(mtry) %>% 
   summarise(acc= mean(acc),
             kappa = mean(kappa)) %>% 
   ungroup()
 
 
 ntree_dat
 
 ggplot(ntree_dat, aes(x=ntree, y = acc)) + geom_point()
 ggplot(mtry_dat, aes(x=mtry, y = acc)) + geom_point()
 
 max(ntree_dat$acc)
 