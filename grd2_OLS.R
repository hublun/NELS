# ####################################################################
# Author: Ashesh Rambachan
# Last updated: Jan 6, 2018
# Description: This script estimates OLS to predict
# grades in college using the full set of covariates.
# I restrict the sample to only blacks and whites. The 
# data are split 80/20 into a training and test set. I estimate
# an OLS predictor that does not have access to race and a
# an OLS that does have access to race on the training
# sample. The test set is used to evaluate the predictive performance
# ####################################################################

library(dplyr)
library(ggplot2)
library(reshape2)

# !ATTENTION!: Set working directory
setwd("~/Dropbox/Harvard/Algorithmic-Bias/NELS-AERPP/ludwig-algorithmic-fairness");

# Loads the cleaned NELS dataset
source("data-code-ludwig/data_clean.R");

y = as.numeric(y) - 1 # y = rec at least GPA >= 2.75
y = 1 - y # y = rec at least GPA < 2.75
blind_covar = subset(covar, select = -c(black, white, y))
black = covar$black
white = covar$white
race_covar = subset(covar, select = -c(white, y))
orthog_covar = race_covar # used to orthogonalize features with respect to race

# Construct interaction terms with black for each covariate
cov_names = colnames(race_covar)
cov_names = cov_names[2:length(cov_names)]
for (i in 1:length(cov_names)) {
  int = race_covar$black * race_covar[[cov_names[i]]]
  race_covar = cbind(race_covar, int)
  colnames(race_covar)[length(colnames(race_covar))] = paste0("black_", cov_names[i], ".Int", sep = "")
}
rm(int, cov_names)

# Construct training and test sets for blind and race aware predictors
set.seed(2000)
train = sample.int(n = nrow(covar), size = floor(0.8*nrow(covar)), replace = F) # index of training set
trg_sz = length(train)
tst_sz = nrow(covar) - trg_sz

black.tst = black[-train]
white.tst = white[-train]
y.tst = y[-train]

# Constructs orthogonal predictors
black.orth = orthog_covar$black
orthog_covar = subset(orthog_covar, select = -c(black))
covar.list = colnames(orthog_covar) 

# loop through each variable, regress on black indicator
# store residuals
for (i in 1:length(covar.list)) {
  resid = lsfit(black.orth, orthog_covar[[covar.list[i]]], intercept = TRUE)$residuals
  orthog_covar[[covar.list[i]]] = resid
}
rm(black.orth, covar.list, resid, i)

#########################
# Estimates regressions #
#########################
blind_covar.ols = blind_covar[, unique(names(blind_covar)), with = FALSE]
blind_ols = lm(y~., blind_covar.ols, subset = train)
blind_covar.tst = blind_covar.ols[-train, ]

aware_covar.ols = race_covar[, unique(names(race_covar)), with = FALSE]
aware_ols = lm(y~., aware_covar.ols, subset = train)
race_covar.tst = aware_covar.ols[-train, ]

orthog_covar.ols = orthog_covar[, unique(names(orthog_covar)), with = FALSE]
orthog_ols = lm(y~., orthog_covar.ols, subset = train)
orthog_covar.tst = orthog_covar.ols[-train, ]

blind_ols.prob = predict(blind_ols, blind_covar.ols[-train, ], type = 'response')
aware_ols.prob = predict(aware_ols, aware_covar.ols[-train, ], type = 'response')
orthog_ols.prob = predict(orthog_ols, orthog_covar.ols[-train, ], type = 'response')

#############
# Heat-maps #
#############
# full sample
raceblindprob.decile = ntile(blind_ols.prob, 10)
raceawareprob.decile = ntile(aware_ols.prob, 10)

heatmap = matrix(rep(0, 10*10), c(10, 10))
for (i in 1:tst_sz) {
  heatmap[raceblindprob.decile[i], raceawareprob.decile[i]] = heatmap[raceblindprob.decile[i], raceawareprob.decile[i]] + 1
}
heatmap = heatmap/tst_sz
dimnames(heatmap) <- list(c('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'), 
                          c('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'))
heatmap.melted <- melt(heatmap)
ggplot(heatmap.melted, aes(x = Var1, y = Var2, fill = value)) + geom_tile(aes(fill=value, col=value)) + 
  ylab('Race aware, deciles of pred. prob') + 
  xlab('Race blind, deciles of pred. prob') + 
  ggtitle("GPA < 2.75: Pred. prob. blind vs. aware") +   
  scale_colour_gradient(low = "white", high = "red") + 
  scale_fill_gradient(low = "white", high = "red")

# whites only
wht.tst = sum(white.tst)
raceblind.prob.w = blind_ols.prob[race_covar.tst$black == 0]
raceaware.prob.w = aware_ols.prob[race_covar.tst$black == 0]

raceblind.dec.w = ntile(raceblind.prob.w, 10)
raceaware.dec.w = ntile(raceaware.prob.w, 10)

heatmap.w = matrix(rep(0, 10*10), c(10, 10))
for (i in 1:wht.tst) {
  heatmap.w[raceblind.dec.w[i], raceaware.dec.w[i]] = heatmap.w[raceblind.dec.w[i], raceaware.dec.w[i]] + 1
}
heatmap.w = heatmap.w/wht.tst
dimnames(heatmap.w) <- list(c('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'), 
                            c('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'))
heatmap.melted.w <- melt(heatmap.w)
ggplot(heatmap.melted.w, aes(x = Var1, y = Var2, fill = value)) + geom_tile(aes(fill=value, col=value)) + 
  ylab('Race aware, deciles of pred. prob') + 
  xlab('Race blind, deciles of pred. prob') + 
  ggtitle("GPA < 2.75: Pred. prob. blind vs. aware, whites") + 
  scale_colour_gradient(low = "white", high = "red") + 
  scale_fill_gradient(low = "white", high = "red")

# blacks only
blk.tst = sum(race_covar.tst$black)
raceblindprob.b = blind_ols.prob[race_covar$black == 1]
raceawareprob.b = aware_ols.prob[race_covar$black == 1]

raceblind.dec.b = ntile(raceblindprob.b, 10)
raceaware.dec.b = ntile(raceawareprob.b, 10)

heatmap.b = matrix(rep(0, 10*10), c(10, 10))
for (i in 1:blk.tst) {
  heatmap.b[raceblind.dec.b[i], raceaware.dec.b[i]] = heatmap.b[raceblind.dec.b[i], raceaware.dec.b[i]] + 1
}
heatmap.b = heatmap.b/blk.tst
dimnames(heatmap.b) <- list(c('B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'), 
                            c('A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10'))
heatmap.melted.b <- melt(heatmap.b)
ggplot(heatmap.melted.b, aes(x = Var1, y = Var2, fill = value)) + geom_tile(aes(fill=value, col=value)) + 
  ylab('Race aware, deciles of pred. prob') + 
  xlab('Race blind, deciles of pred. prob') + 
  scale_colour_gradient(low = "white", high = "red") + 
  scale_fill_gradient(low = "white", high = "red") + 
  ggtitle("GPA < 2.75: Pred. prob. blind vs. aware, black") 

#####################################################################
# Predictor curve: Fixes number of individuals admitted. Varies the #
# fraction of admits that are African American and plots the        #
# associated efficiency for each predictor                          #
#####################################################################
# Admits 50% of test sample
total.n = 483
frac_blk = seq(from = 0.01, to = 0.2, by = 0.01) # target
frac_blk.obs = matrix(0, nrow = length(frac_blk), ncol = 1) # observed
admit.eff = matrix(0, nrow = length(frac_blk), ncol = 3)

# Sorted order: Blind
wht_blindprob.srt = sort(blind_ols.prob[white.tst == 1], decreasing = FALSE)
blk_blindprob.srt = sort(blind_ols.prob[black.tst == 1], decreasing = FALSE)

# Sorted order: Orthogonal
wht_orthogprob.srt = sort(orthog_ols.prob[white.tst == 1], decreasing = FALSE)
blk_orthogprob.srt = sort(orthog_ols.prob[black.tst == 1], decreasing = FALSE)

# Sorted order: Aware
wht_awareprob.srt = sort(aware_ols.prob[white.tst == 1], decreasing = FALSE)
blk_awareprob.srt = sort(aware_ols.prob[black.tst == 1], decreasing = FALSE)

# Loops through each fraction of admits that are black
for (i in 1:length(frac_blk)) {
  blk.n = ceiling(frac_blk[i] * total.n)
  wht.n = total.n - blk.n
  
  # race blind admit rule
  whiteblind_cutoff = wht_blindprob.srt[wht.n]
  blackblind_cutoff = blk_blindprob.srt[blk.n]
  
  blind_admit = ifelse(white.tst == 1 & blind_ols.prob <= whiteblind_cutoff |
                         black.tst == 1 & blind_ols.prob <= blackblind_cutoff, 1, 0)
  blind_admit.wht = sum(white.tst[blind_admit == 1])
  blind_admit.blk = sum(black.tst[blind_admit == 1])
  
  # used for debugging
  #print(paste0("Blind admit rule admits: ", sum(blind_admit), " individuals"))
  #print(paste0("Blind admit rule admits: ", blind_admit.wht/total.n, " whites"))
  #print(paste0("Blind admit rule admits ", blind_admit.blk/total.n, " blacks"))
  
  # race orthog admit rule
  whiteorthog_cutoff = wht_orthogprob.srt[wht.n]
  blackorthog_cutoff = blk_orthogprob.srt[blk.n]
  
  orthog_admit = ifelse(white.tst == 1 & orthog_ols.prob <= whiteorthog_cutoff |
                          black.tst == 1 & orthog_ols.prob <= blackorthog_cutoff, 1, 0)
  orthog_admit.wht = sum(white.tst[orthog_admit == 1])
  orthog_admit.blk = sum(black.tst[orthog_admit == 1])
  
  # used for debugging
  #print(paste0("Orthog admit rule admits ", sum(orthog_admit), " individuals"))
  #print(paste0("Orthog admit rule admits ", orthog_admit.wht/total.n, " whites"))
  #print(paste0("Orthog admit rule admits ", orthog_admit.blk/total.n, " blacks"))
  
  # race aware admit rule
  whiteaware_cutoff = wht_awareprob.srt[wht.n]
  blackaware_cutoff = blk_awareprob.srt[blk.n]
  
  aware_admit = ifelse(white.tst == 1 & aware_ols.prob <= whiteaware_cutoff |
                         black.tst == 1 & aware_ols.prob <= blackaware_cutoff, 1, 0)
  aware_admit.wht = sum(white.tst[aware_admit == 1])
  aware_admit.blk = sum(black.tst[aware_admit == 1])
  
  # used for debugging
  #print(paste0("Aware admit rule admits ", sum(aware_admit), " individuals"))
  #print(paste0("Aware admit rule admits ", aware_admit.wht/total.n, " whites"))
  #print(paste0("Aware admit rule admits ", aware_admit.blk/total.n, " blacks"))
  
  # Accuracy:
  admit.eff[i, 1] = mean(y.tst[blind_admit == 1])
  admit.eff[i, 2] = mean(y.tst[orthog_admit == 1])
  admit.eff[i, 3] = mean(y.tst[aware_admit == 1])
  
  # Obs Black Frac
  frac_blk.obs[i] = blk.n/total.n
  
  # used for debugging
  #print(paste0('-------------------'))
}
frac_blk.obs = frac_blk.obs * 100
admit.eff = admit.eff * 100

# Constructs the admits of the efficient planner
rule = 50
blind_ols.pctile = ntile(blind_ols.prob, 100)
orthog_ols.pctile = ntile(orthog_ols.prob, 100)
aware_ols.pctile = ntile(aware_ols.prob, 100)

blind_admit.effp = ifelse(blind_ols.pctile <= rule, 1, 0)
orthog_admit.effp = ifelse(orthog_ols.pctile <= rule, 1, 0)
aware_admit.effp = ifelse(aware_ols.pctile <= rule, 1, 0)

effp.fracblk = matrix(0, nrow = 3, ncol = 1)
effp.acc = matrix(0, nrow = 3, ncol = 1)

effp.acc[1] = mean(y.tst[blind_admit.effp == 1])
effp.acc[2] = mean(y.tst[orthog_admit.effp == 1])
effp.acc[3] = mean(y.tst[aware_admit.effp == 1])
effp.acc = effp.acc*100

effp.fracblk[1] = mean(black.tst[blind_admit.effp == 1])
effp.fracblk[2] = mean(black.tst[orthog_admit.effp == 1])
effp.fracblk[3]  = mean(black.tst[aware_admit.effp == 1])
effp.fracblk = effp.fracblk*100

plot(frac_blk.obs[5:20], admit.eff[5:20, 1], type = 'l', lty = 3, col = 'black',
     main = 'Error rate as % of black admits varies',
     xlab = '% of admits that are black', ylab = "% Not rec. at least mostly B's", 
     ylim = c(12, 17)
)
lines(frac_blk.obs[5:20], admit.eff[5:20, 2], type = 'l', lty = 2,  col = 'red')
lines(frac_blk.obs[5:20], admit.eff[5:20, 3], type = 'l', lty = 1, col = 'blue')
points(effp.fracblk[1], effp.acc[1], type = 'p', pch = 1, col = 'black')
points(effp.fracblk[2], effp.acc[2], type = 'p', pch = 2, col = 'black')
points(effp.fracblk[3], effp.acc[3], type = 'p', pch = 3, col = 'black')
legend('bottomright', legend = c('blind', 'orthog.', 'aware', 
                                 'eff. blind', 'eff. orthog.', 'eff. aware'), 
       lty = c(3, 2, 1, NA, NA, NA), pch = c(NA, NA, NA, 1, 2, 3),  
       col = c('black', 'red', 'blue', 'black', 'black', 'black'), 
       pt.cex = 1, cex = 0.6)
