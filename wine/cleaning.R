library(randomForest)
library(tidyverse)
library(GGally)
library(skimr)
library(corrplot)
library(glmnet)
library(caret)
library(nnet)
library(ncvreg)
library(kernlab)
library(gbm)

source("func.R")

red <- read_delim("winequality-red.csv", delim = ";")
red <- read_delim("winequality-white.csv", delim = ";")
red$quality <- ifelse(red$quality <= 4, 1, ifelse(red$quality >=5 & red$quality <= 6, 2, 3))

red$quality <- factor(red$quality)

ggplot(red, aes(quality)) + geom_histogram(stat="count") +
  theme_tufte(11) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") +
  xlab("Quality") +
  ylab("Observations")
ggsave("figures/quality_histogram.pdf", width = 6, height = 4)
# red <- read_delim("winequality-white.csv", delim = ";")
red2 <- gather(red, key = variable, value = value, -quality)
red <- red %>% drop_na() 

colnames(red) <- red %>% colnames() %>% str_replace_all(" ","_")
dat <- red

red <- dplyr::select(red, -residual_sugar, -free_sulfur_dioxide, -total_sulfur_dioxide, -chlorides, density, pH)

#red <- select(red, -residual_sugar, fixed_acidity, chlorides, citric_acid)
# Principle Component Analysis
pca_dat <- dplyr::select(red, -quality)
pca <- prcomp(pca_dat, scale = TRUE)
plot(pca)
summary(pca)
summary(pca)[3]
pr.var=pca$sdev ^2
pve=pr.var/sum(pr.var)
pve - lead(pve)
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained ", type='b')
ggsave("figures/pca_var_exp.pdf", width = 6, height = 4)

plot(cumsum(pve), xlab="Principal Component ", ylab=" Cumulative Proportion of Variance Explained", ylim=c(0,1), type='b')
ggsave("figures/pca_cum_var_exp.pdf", width = 6, height = 4)

ggbiplot(pca, obs.scale = 1, var.scale = 1,
  groups = dat$quality, ellipse = TRUE, circle = TRUE) +
  scale_color_discrete(name = '') +
  theme(legend.direction = 'horizontal', legend.position = 'top') +
  theme_tufte(11) +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey")

ggsave("figures/biplot.pdf", width = 10, height = 10)

pcadat <- data.frame(quality = dat$quality)
pcadat[, 2:9] <- pca$x[, 1:8]
names(pcadat) <- c("quality", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8")
saveRDS(pcadat, "data/pca_red.Rds")


biplot(pca)

dat <- data.frame(quality = red$quality)
dat <- cbind(dat, pca$x[, 1:5])
head(dat)

# Variable Selection
# ------------------
# Alcohol: Preference for more alcohol
# Chloride: salt taste. Higher chloride is preferred less
# Citric acid: preservative; taste is tart or sharp, preferred. Can add ‘freshness’ and flavor to wines
# Fixed-acidity: fruit acid of the grape; 
# Sulphates: additive; antimicrobial and antioxidant
# Volatile acidity: Higher levels produce vinegar taste

# Remove
# ------------------
# Density: "heaviness" of wine. (range = 0.9901 to 1.0037). Not enough to see preference
# pH:  acidic or basic a wine; generally between 3-4; no preference
# Total sulfur dioxide: evident in the nose, generally untasteable except at higher levels
# Residual sugar: left over after fermintation; should be low; produces sweetness
# Free sulfur Dioxide: Additive; prevents microbial growth and the oxidation of wine



X_train <- as.data.frame(dplyr::select(dat, -quality))
y_train <- as.data.frame(dplyr::select(dat, quality))

# Lasso regression
cv.out <- cv.glmnet(as.matrix(X_train), as.matrix(y_train$quality), alpha=1, family="multinomial")
summary(cv.out)
lambda_min <- cv.out$lambda.min
#best value of lambda
lambda_1se <- cv.out$lambda.1se
#regression coefficients
coef(cv.out,s=lambda_1se)

mod <- cv.ncvreg(X_train, as.factor(y_train$quality), family = "poisson")
coef(mod)
plot(mod)

# (Intercept)          -5.59984822
# fixed_acidity         0.03264552
# volatile_acidity     -2.33257497
# citric_acid           0.16515787
# residual_sugar        .         
# chlorides             .         
# free_sulfur_dioxide   .         
# total_sulfur_dioxide  .         
# density               .         
# pH                    .         
# sulphates             1.69832643
# alcohol               0.80041295

#          (Intercept)        fixed_acidity     volatile_acidity 
#           0.97569358           0.00000000          -0.28540808 
#          citric_acid       residual_sugar            chlorides 
#           0.00000000           0.00000000          -0.55534142 
#  free_sulfur_dioxide total_sulfur_dioxide              density 
#           0.00154689          -0.00103667           0.00000000 
#                   pH            sulphates              alcohol 
#          -0.12422717           0.23711409           0.07529040 

# Residual sugar, free sulfur dioxide, fixed acidity, residual sugar

# Feature Importance
control <- trainControl(method="repeatedcv",number=10, repeats = 10)
model <- train(quality ~ ., data = red, method = "rf", preProcess = c("scale", "center"), trControl = control)

importance<- varImp(model, scale=T)
imp_df1 <- importance$importance
imp_df1$group <- rownames(imp_df1)

imp_df1 %>%
  ggplot(aes(x=reorder(group,Overall),y=Overall),size=2) +
  theme_tufte(11) +
  geom_bar(stat = "identity") +
  annotate("segment", x=-Inf, xend=Inf, y=-Inf, yend=-Inf, color = "grey") +
  annotate("segment", x=-Inf, xend=-Inf, y=-Inf, yend=Inf, color = "grey") +
  theme(axis.text.x = element_text(vjust=1,angle=90))+
  labs(x="Variable",y="Overall Importance",title="Scaled Feature Importance")
ggsave("figures/rf_cv_important_variables.pdf", width = 6, height = 4)
# Recursive Feature Elimination using Random Forest
control <- rfeControl(functions=rfFuncs, method = "repeatedcv", number=10, repeats=10)
results_1 <- rfe(x=X_train, y=y_train$quality, rfeControl = control)
results_1
predictors(results_1)



# Save
saveRDS(dat, "data/red_clean.Rds")

saveRDS(dat, "data/red_clean_three_groups.Rds")

