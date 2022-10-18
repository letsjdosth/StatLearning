# HW1 Problem5

library(SemiPar)
library(splines)
library(gam)
library(mgcv)

data(fossil)
head(fossil)
plot(strontium.ratio ~ age, data=fossil)

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
# lm3_fit_pred
plot(strontium.ratio ~ age, data=fossil)
lines(lm3_fit_pred[,1]~pred_df_age$age, col="blue")
lines(lm3_fit_pred[,2]~pred_df_age$age)
lines(lm3_fit_pred[,3]~pred_df_age$age)

pred_df_age2 = data.frame(age=c(95, 115))
lm3_fit_pred2 = predict(lm3_fit, pred_df_age2, interval="prediction")
print(lm3_fit_pred2)


# cubic spline fit (without smoothing penalty)
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

# using gcv, cubic spline fit
candid_knot_num = 2:20
gcv_for_candid_knot_num = rep(0, length(candid_knot_num))
for(i in 1:length(candid_knot_num)){
    knot_num = candid_knot_num[i]
    knot_position = (125-95)/(knot_num+2)*1:knot_num+95
    lm_ns_fit = lm(strontium.ratio ~ ns(age, knots=knot_position), data=fossil)
    res = sum(lm_ns_fit$residuals^2)
    trS = sum(lm.influence(lm_ns_fit)$hat)
    gcv_for_candid_knot_num[i] = res/(1-trS/length(fossil$age))^2
}

(knot_num = candid_knot_num[which.min(gcv_for_candid_knot_num)]) #15
knot_position = (125-95)/(knot_num+2)*1:knot_num+95
lm_ns15_fit = lm(strontium.ratio ~ ns(age, knots=knot_position), data=fossil)
summary(lm_ns15_fit)
plot(strontium.ratio ~ age, data=fossil)
lm_ns15_fit_pred = predict(lm_ns15_fit, pred_df_age, interval="prediction")
lines(lm_ns15_fit_pred[,1]~pred_df_age$age, col="blue")
lines(lm_ns15_fit_pred[,2]~pred_df_age$age)
lines(lm_ns15_fit_pred[,3]~pred_df_age$age)

lm_ns15_fit_pred2 = predict(lm_ns15_fit, pred_df_age2, interval="prediction")
print(lm_ns15_fit_pred2)





# smoothing spline
ss_gam_fit = gam(strontium.ratio~s(age, bs="cr"), data=fossil)
ss_gam_fit #automatically use gcv?

ss_gam_fit_pred = predict(ss_gam_fit, pred_df_age, interval="prediction", se.fit=TRUE)
# ss_gam_fit_pred
# ?predict.gam
plot(strontium.ratio ~ age, data=fossil)
lines(ss_gam_fit_pred$fit~pred_df_age$age, col="blue")
lines(ss_gam_fit_pred$fit + 1.96*ss_gam_fit_pred$se.fit ~ pred_df_age$age)
lines(ss_gam_fit_pred$fit - 1.96*ss_gam_fit_pred$se.fit ~ pred_df_age$age)

ss_gam_fit_pred2 = predict(ss_gam_fit, pred_df_age2, interval="prediction", se.fit=TRUE)
ss_gam_fit_pred2 
data.frame(fit=ss_gam_fit_pred2$fit,
            lwr=ss_gam_fit_pred2$fit - 1.96*ss_gam_fit_pred2$se.fit,
            uwr=ss_gam_fit_pred2$fit + 1.96*ss_gam_fit_pred2$se.fit)
#right? need to check


# local regression
loess_fit75 = loess(strontium.ratio ~ age, data=fossil, span=0.75) #chosen by my hands
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
(candid_span[which.min(gcv_for_candid_span)])
#but span=0.24 gives error

loess_fit = loess(strontium.ratio ~ age, data=fossil, span=candid_span[which.min(gcv_for_candid_span)])
loess_fit_pred = predict(loess_fit, pred_df_age, interval="prediction", se=TRUE)
loess_fit_pred
plot(strontium.ratio ~ age, data=fossil)
lines(loess_fit_pred$fit~pred_df_age$age, col="blue")
lines(loess_fit_pred$fit + qt(0.975, loess_fit_pred$df)*loess_fit_pred$se.fit ~ pred_df_age$age)
lines(loess_fit_pred$fit - qt(0.975, loess_fit_pred$df)*loess_fit_pred$se.fit ~ pred_df_age$age) 

loess_fit_pred2 = predict(loess_fit, pred_df_age2, interval="prediction", se=TRUE)
loess_fit_pred2
data.frame(fit=loess_fit_pred2$fit,
            lwr=loess_fit_pred2$fit - 1.96*loess_fit_pred2$se.fit,
            uwr=loess_fit_pred2$fit + 1.96*loess_fit_pred2$se.fit)
