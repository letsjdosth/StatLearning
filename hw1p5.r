# HW1 Problem5

library(SemiPar)
library(splines)
library(gam)
library(mgcv)

data(fossil)
head(fossil)
plot(strontium.ratio ~ age, data=fossil)

#========================================================================
#polynomial regression
lm1_fit = lm(strontium.ratio ~ age, data=fossil)
lm2_fit = lm(strontium.ratio ~ age+ I(age^2), data=fossil)
lm3_fit = lm(strontium.ratio ~ age + I(age^2) + I(age^3), data=fossil)
lm4_fit = lm(strontium.ratio ~ age + I(age^2) + I(age^3) + I(age^4), data=fossil)

plot(strontium.ratio ~ age, data=fossil)
curve(coef(lm1_fit)[1]+coef(lm1_fit)[2]*x, col="black", add=TRUE)
curve(coef(lm2_fit)[1]+coef(lm2_fit)[2]*x+coef(lm2_fit)[3]*x^2, col="blue", add=TRUE)
curve(coef(lm3_fit)[1]+coef(lm3_fit)[2]*x+coef(lm3_fit)[3]*x^2+coef(lm3_fit)[4]*x^3, col="red", add=TRUE)
curve(coef(lm4_fit)[1]+coef(lm4_fit)[2]*x+coef(lm4_fit)[3]*x^2+coef(lm4_fit)[4]*x^3+coef(lm4_fit)[5]*x^4, col="green", add=TRUE)

#order 3
pred_df_age = data.frame(age=90:125)
lm3_fit_pred = predict(lm3_fit, pred_df_age, interval="prediction")
plot(strontium.ratio ~ age, data=fossil)
lines(lm3_fit_pred[,1]~pred_df_age$age, col="blue")
lines(lm3_fit_pred[,2]~pred_df_age$age)
lines(lm3_fit_pred[,3]~pred_df_age$age)

pred_df_age2 = data.frame(age=c(95, 115))
lm3_fit_pred2 = predict(lm3_fit, pred_df_age2, interval="prediction")
print(lm3_fit_pred2)

#========================================================================
# cubic spline fit
knot_num = 2 #using my eye
knot_position = (125-95)/(knot_num+2)*1:knot_num+95
lm_ns2_fit = lm(strontium.ratio ~ ns(age, knots=knot_position), data=fossil)
summary(lm_ns2_fit)
plot(strontium.ratio ~ age, data=fossil)
lm_ns2_fit_pred = predict(lm_ns2_fit, pred_df_age, interval="prediction")
lines(lm_ns2_fit_pred[,1]~pred_df_age$age, col="blue")
lines(lm_ns2_fit_pred[,2]~pred_df_age$age)
lines(lm_ns2_fit_pred[,3]~pred_df_age$age)

lm_ns2_fit_pred2 = predict(lm3_fit, pred_df_age2, interval="prediction")
print(lm_ns2_fit_pred2)

# names(lm_ns2_fit) # all properties
# class(lm_ns2_fit) #class name
# showMethods(class="lm") #for S4
# methods(class="lm") #for S3
lm_ns2_X = model.matrix(lm_ns2_fit)
lm_ns2_smoother = lm_ns2_X %*% solve(t(lm_ns2_X)%*%lm_ns2_X) %*% t(lm_ns2_X)
lm_ns2_smoother_diag = diag(lm_ns2_smoother)

# using loocv & GCV
candid_knot_num = 2:20
loocv_for_candid_knot_num = rep(0, length(candid_knot_num))
gcv_for_candid_knot_num = rep(0, length(candid_knot_num))
for(i in 1:length(candid_knot_num)){
    knot_num = candid_knot_num[i]
    knot_position = (125-95)/(knot_num+2)*1:knot_num+95
    lm_ns_fit = lm(strontium.ratio ~ ns(age, knots=knot_position), data=fossil)

    lm_ns_X = model.matrix(lm_ns_fit)
    lm_ns_smoother = lm_ns_X %*% solve(t(lm_ns_X)%*%lm_ns_X) %*% t(lm_ns_X)
    lm_ns_smoother_diag = diag(lm_ns_smoother)
    lm_ns_smoother_trace = sum(lm_ns_smoother_diag)

    lm_ns_loocv = 0
    for(j in 1:length(lm_ns_smoother_diag)){
        lm_ns_loocv = lm_ns_loocv + (lm_ns_fit$residuals[j]/(1 - lm_ns_smoother_diag[j]))^2
    }
    lm_ns_gcv = sum(lm_ns_fit$residuals^2) / (1 - lm_ns_smoother_trace/length(lm_ns_smoother_diag))^2
    loocv_for_candid_knot_num[i] = lm_ns_loocv/length(lm_ns_smoother_diag)
    gcv_for_candid_knot_num[i] = lm_ns_gcv/length(lm_ns_smoother_diag)
}

(candid_knot_num[which.min(loocv_for_candid_knot_num)]) #15
(candid_knot_num[which.min(gcv_for_candid_knot_num)]) #15

(knot_num = candid_knot_num[which.min(gcv_for_candid_knot_num)])
knot_position = (125-95)/(knot_num+2)*1:knot_num+95
lm_ns_cvknot_fit = lm(strontium.ratio ~ ns(age, knots=knot_position), data=fossil)
summary(lm_ns_cvknot_fit)
plot(strontium.ratio ~ age, data=fossil)
lm_ns_cvknot_fit_pred = predict(lm_ns_cvknot_fit, pred_df_age, interval="prediction")
lines(lm_ns_cvknot_fit_pred[,1]~pred_df_age$age, col="blue")
lines(lm_ns_cvknot_fit_pred[,2]~pred_df_age$age)
lines(lm_ns_cvknot_fit_pred[,3]~pred_df_age$age)

lm_ns_cvknot_fit_pred2 = predict(lm_ns_cvknot_fit, pred_df_age2, interval="prediction")
print(lm_ns_cvknot_fit_pred2)


#========================================================================
# smoothing spline

## using smooth.spline
candid_df = 5:20
ss_loocv_vec = rep(0, length(candid_df))
ss_gcv_vec = rep(0, length(candid_df))

for(i in 1:length(candid_df)){
    ss_fit_loocv = smooth.spline(fossil$age, fossil$strontium.ratio, nknots=candid_df[i], cv=TRUE)
    ss_loocv_vec[i] = ss_fit_loocv$cv.crit
    ss_fit_gcv = smooth.spline(fossil$age, fossil$strontium.ratio, nknots=candid_df[i], cv=FALSE)
    ss_gcv_vec[i] = ss_fit_gcv$cv.crit
}
(candid_df[which.min(ss_loocv_vec)]) #12
(candid_df[which.min(ss_gcv_vec)]) #12

ss_fit_loocv = smooth.spline(fossil$age, fossil$strontium.ratio, nknots=candid_df[which.min(ss_loocv_vec)], cv=TRUE)
ss_fit_gcv = smooth.spline(fossil$age, fossil$strontium.ratio, nknots=candid_df[which.min(ss_gcv_vec)], cv=FALSE)
sum(ss_fit_gcv$lev) #eff number of parameters
ss_fit_loocv_pred = predict(ss_fit_loocv, pred_df_age, interval="prediction", se=TRUE) #no se output
ss_fit_gcv_pred = predict(ss_fit_gcv, pred_df_age, interval="prediction", se=TRUE) #no se output

plot(strontium.ratio ~ age, data=fossil)
lines(ss_fit_loocv_pred$y[,1] ~ ss_fit_loocv_pred$x[,1], col="blue")
lines(ss_fit_gcv_pred$y[,1] ~ ss_fit_gcv_pred$x[,1], col="red")
#exactly the same.


# using gam
ss_gam_fit = gam(strontium.ratio~bs(age, bs="cr"), data=fossil)
print(ss_gam_fit) #automatically use gcv. #k=10
# print(gam(strontium.ratio~s(age, bs="cr", k=10), data=fossil))

ss_gam_fit_pred = predict(ss_gam_fit, pred_df_age, interval="prediction", se.fit=TRUE)
# plot(strontium.ratio ~ age, data=fossil)
lines(ss_gam_fit_pred$fit~pred_df_age$age, col="green") #different!
lines(ss_gam_fit_pred$fit + 1.96*ss_gam_fit_pred$se.fit ~ pred_df_age$age)
lines(ss_gam_fit_pred$fit - 1.96*ss_gam_fit_pred$se.fit ~ pred_df_age$age)

ss_gam_fit_pred2 = predict(ss_gam_fit, pred_df_age2, interval="prediction", se.fit=TRUE)
ss_gam_fit_pred2
data.frame(fit=ss_gam_fit_pred2$fit,
            lwr=ss_gam_fit_pred2$fit - 1.96*ss_gam_fit_pred2$se.fit,
            uwr=ss_gam_fit_pred2$fit + 1.96*ss_gam_fit_pred2$se.fit)



#========================================================================
# local regression
loess_fit75 = loess(strontium.ratio ~ age, data=fossil, span=0.75) #chosen by my hands
class(loess_fit75)
methods(class="loess")
names(loess_fit75)

loess_fit75_pred = predict(loess_fit75, pred_df_age, interval="prediction", se=TRUE)
loess_fit75_pred
plot(strontium.ratio ~ age, data=fossil)
lines(loess_fit75_pred$fit~pred_df_age$age, col="blue")
lines(loess_fit75_pred$fit + qt(0.975, loess_fit75_pred$df)*loess_fit75_pred$se.fit ~ pred_df_age$age)
lines(loess_fit75_pred$fit - qt(0.975, loess_fit75_pred$df)*loess_fit75_pred$se.fit ~ pred_df_age$age) 


#using gcv, loess
candid_span = seq(0.1, 2, 0.01)
gcv_for_candid_span = rep(0, length(candid_span))
for(i in 1:length(candid_span)){
    N = length(fossil$age)
    loess_fit = loess(strontium.ratio ~ age, data=fossil, span=candid_span[i])
    GCV = sum(loess_fit$residuals^2) / (1 -loess_fit$trace.hat/N)^2
    gcv_for_candid_span[i] = GCV
}
(candid_span[which.min(gcv_for_candid_span)]) #0.24

loess_fit = loess(strontium.ratio ~ age, data=fossil, span=candid_span[which.min(gcv_for_candid_span)])
loess_fit_pred = predict(loess_fit, pred_df_age, interval="prediction", se=TRUE)
loess_fit_pred
plot(strontium.ratio ~ age, data=fossil)
lines(loess_fit_pred$fit~pred_df_age$age, col="blue")
lines(loess_fit_pred$fit + qt(0.975, loess_fit_pred$df)*loess_fit_pred$se.fit ~ pred_df_age$age)
lines(loess_fit_pred$fit - qt(0.975, loess_fit_pred$df)*loess_fit_pred$se.fit ~ pred_df_age$age) 

loess_fit_pred2 = predict(loess_fit, pred_df_age2, interval="prediction", se=TRUE)
# loess_fit_pred2
data.frame(fit=loess_fit_pred2$fit,
            lwr=loess_fit_pred2$fit - 1.96*loess_fit_pred2$se.fit,
            uwr=loess_fit_pred2$fit + 1.96*loess_fit_pred2$se.fit)
