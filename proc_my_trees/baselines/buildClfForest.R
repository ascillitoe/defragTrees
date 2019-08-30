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
y <- as.factor(Z[,ncol(Z)])
z <- Z[,ncol(Z)]
Z2 <- read.csv(test_file, header=F)
X2 <- Z2[,1:(ncol(Z2)-1)]
y2 <- as.factor(Z2[,ncol(Z2)])
z2 <- Z2[,ncol(Z2)]

# rf training
#nsize <- c(10, 50, 100, 150)
#msize <- c(16, 32, 64, 128)
#rf <- randomForest(X, y, ntree=ntree, nodesize=nsize[1], maxnodes=msize[1])
#ns <- nsize[1]
#ms <- msize[1]
#err = rf$err.rate[ntree]
#for (nstmp in nsize) {
#    for (mstmp in msize) {
#        rftmp <- randomForest(X, y, ntree=ntree, nodesize=nstmp, maxnodes=mstmp)
#        errtmp <- rftmp$err.rate[ntree]
#        cat("ns = ", nstmp, "ms = ", mstmp, "err = ", errtmp, "\n")
#        if (err > rftmp$err.rate[ntree]) {
#            err <- rftmp$err.rate[ntree]
#            rf <- rftmp
#            ns <- nstmp
#            ms <- mstmp
#        }
#    }
#}
#print(ns)
#print(ms)
#print(err)
rf <- randomForest(X, y, ntree=ntree, nodesize=50, maxnodes=64)

# inTrees
print("inTrees: Creating treeList with RF2List")
treeList <- RF2List(rf)
print("inTrees: Extracting unique rules")
ruleExec <- unique(extractRules(treeList, X))
print("inTrees: Getting rule metric")
ruleMetric <- getRuleMetric(ruleExec, X, y)
print("inTrees: Pruning rules")
ruleMetric <- pruneRule(ruleMetric, X, y)
print("inTrees: Building learner")
learner <- buildLearner(ruleMetric, X, y)
print("inTrees: Capturing learners output")
out <- capture.output(learner)

# nodeHarvest
print("nodeHarvest: Running")
nh <- nodeHarvest(X, z)
print("nodeHarvest: Capturing output")
out2 <- capture.output(nh$nodes)

# save
print("RF: Training prediction")
z1 <- predict(rf, X)
print("RF: Test prediction")
z2 <- predict(rf, X2)
print("NH: Training prediction")
z3 <- as.numeric(predict(nh, X) > 0.5)
print("NH: Test prediction")
z4 <- as.numeric(predict(nh, X2) > 0.5)

print("Saving buildClfForest.R results")
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
