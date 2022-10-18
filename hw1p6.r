# HW1 prob6

# loading data
calif <- read.table("https://raw.githubusercontent.com/jbryer/CompStats/master/Data/cadata.dat",header = TRUE)
calif$MedianOver <- factor(as.numeric(calif$MedianHouseValue > 250000))
cut_calif <- calif[c("MedianOver", "Latitude", "Longitude")]
plot(Latitude ~ Longitude, data=cut_calif, col=cut_calif$MedianOver)

#for fitting check
grid <- data.frame(
    Longitude = rep(seq(min(cut_calif$Longitude), max(cut_calif$Longitude), length.out=100), each=100),
    Latitude = rep(seq(min(cut_calif$Latitude), max(cut_calif$Latitude), length.out=100), times=100)
)


# SVM
library(e1071)
# ?svm
#cost: picked from 1 to 13
# cost of constraints violation (default: 1)
# —it is the ‘C’-constant of the regularization term in the Lagrange formulation.
svm_fit = svm(MedianOver~., data=cut_calif, kernel="linear", scale=FALSE, cost=6)
plot(svm_fit, cut_calif)
summary(svm_fit)

svm_fit_y = predict(svm_fit, cut_calif[-1])
svm_fit_false_rate = sum(cut_calif$MedianOver!=svm_fit_y) / length(cut_calif$MedianOver)
print(1 - svm_fit_false_rate) #prediction accuracy
# c=6, [1] 0.7584302

# CART
library(rpart)
# ?rpart
cart_fit = rpart(MedianOver~., data=cut_calif, method="class")
printcp(cart_fit)
summary(cart_fit)
plot(cart_fit, uniform=TRUE, main="Classification Tree")
text(cart_fit, use.n=TRUE, all=TRUE, cex=.8)

pruned_cart_fit = prune(cart_fit, cp=cart_fit$cptable[which.min(cart_fit$cptable[,"xerror"]),"CP"])
plot(pruned_cart_fit, uniform=TRUE, main="Pruned Classification Tree")
text(pruned_cart_fit, use.n=TRUE, all=TRUE, cex=.8)
print(pruned_cart_fit, digit=2)
pruned_cart_fit$cptable

grid$pruned_cart_fit_on_grid = predict(pruned_cart_fit, grid, "class")
plot(grid$Longitude, grid$Latitude, col=grid$pruned_cart_fit_on_grid, pch=1)
points(Latitude ~ Longitude, data=cut_calif, col=cut_calif$MedianOver)

cart_fit_y = predict(pruned_cart_fit, cut_calif[-1], "class")
cart_fit_false_rate = sum(cut_calif$MedianOver!=cart_fit_y)/length(cut_calif$MedianOver)
print(1 - cart_fit_false_rate)



#random forest
library(randomForest)
# ?randomForest
set.seed(20221017)
rf_fit = randomForest(MedianOver~., data=cut_calif, proximity=TRUE) #super time consuming, memory burst!!
print(rf_fit)

grid$rf_fit_on_grid = predict(rf_fit, grid)
plot(grid$Longitude, grid$Latitude, col=grid$rf_fit_on_grid, pch=1)
points(Latitude ~ Longitude, data=cut_calif, col=cut_calif$MedianOver)

rf_fit_y = predict(rf_fit, cut_calif[-1])
rf_fit_false_rate = sum(cut_calif$MedianOver!=rf_fit_y)/length(cut_calif$MedianOver)
print(1 - rf_fit_false_rate)
