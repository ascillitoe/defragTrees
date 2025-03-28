# args
train_file = commandArgs(trailingOnly=TRUE)[1]
test_file = commandArgs(trailingOnly=TRUE)[2]
res_dir = commandArgs(trailingOnly=TRUE)[3]
ntree = commandArgs(trailingOnly=TRUE)[4]
seed = commandArgs(trailingOnly=TRUE)[5]
ntree <- as.integer(ntree)
seed <- as.integer(seed)

# library
library(randomForest)
library(inTrees)
library(nodeHarvest)
set.seed(seed)

# data
Z <- read.csv(train_file, header=F)
X <- Z[,1:(ncol(Z)-1)]
y <- Z[,ncol(Z)]
z <- Z[,ncol(Z)]
Z2 <- read.csv(test_file, header=F)
X2 <- Z2[,1:(ncol(Z2)-1)]
y2 <- Z2[,ncol(Z2)]
z2 <- Z2[,ncol(Z2)]

# rf training
nsize <- c(10, 50, 100, 150)
msize <- c(16, 32, 64, 128)
rf <- randomForest(X, y, ntree=ntree, nodesize=nsize[1], maxnodes=msize[1])
ns <- nsize[1]
ms <- msize[1]
err = rf$mse[ntree]
for (nstmp in nsize) {
    for (mstmp in msize) {
        rftmp <- randomForest(X, y, ntree=ntree, nodesize=nstmp, maxnodes=mstmp)
        errtmp <- rftmp$mse[ntree]
        cat("ns = ", nstmp, "ms = ", mstmp, "err = ", errtmp, "\n")
        if (err > rftmp$mse[ntree]) {
            err <- rftmp$mse[ntree]
            rf <- rftmp
            ns <- nstmp
            ms <- mstmp
        }
    }
}
print(ns)
print(ms)
print(err)

# inTrees
treeList <- RF2List(rf)
ruleExec <- unique(extractRules(treeList, X))
ruleMetric <- getRuleMetric(ruleExec, X, dicretizeVector(y))
ruleMetric <- pruneRule(ruleMetric, X, dicretizeVector(y))
learner <- buildLearner(ruleMetric, X, dicretizeVector(y))
out <- capture.output(learner)

# nodeHarvest
nh <- nodeHarvest(X, z)
out2 <- capture.output(nh$nodes)

# save
z1 <- predict(rf, X)
z2 <- predict(rf, X2)
z3 <- predict(nh, X)
z4 <- predict(nh, X2)
forest_dir = sprintf('%s/forest', res_dir)
dir.create(res_dir, showWarnings = FALSE)
dir.create(forest_dir, showWarnings = FALSE)
write.table(z1, sprintf('%s/pred_train.csv', res_dir), quote=F, col.names=F, append=F)
write.table(z2, sprintf('%s/pred_test.csv', res_dir), quote=F, col.names=F, append=F)
write.table(z3, sprintf('%s/pred_train_nh.csv', res_dir), quote=F, col.names=F, append=F)
write.table(z4, sprintf('%s/pred_test_nh.csv', res_dir), quote=F, col.names=F, append=F)
cat(out,file=sprintf("%s/inTrees.txt", res_dir, t),sep="\n",append=FALSE)
cat(out2,file=sprintf("%s/nodeHarvest.txt", res_dir, t),sep="\n",append=FALSE)
for (t in 1:ntree) {
    out <- capture.output(getTree(rf, k=t))
    cat(out,file=sprintf("%s/tree%03d.txt", forest_dir, t),sep="\n",append=FALSE)
}
