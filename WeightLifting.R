
library(caret)
library(randomForest)

path_to_data <- "C:/Users/Joel/Documents/R Scripts/Practical Machine Learning/Course Project Writeup" 

coerce_numeric <- function(df) {
	classes <- sapply(df[1,], class)
	keep_as_char <- c("classe", "user_name", "cvtd_timestamp", "new_window")
	(!names(classes) %in% keep_as_char) & classes=="character"
	classes
}

readDataset <- function(train_or_test) {
	ds <- read.csv(
		paste(path_to_data, "/pml-", train_or_test, "ing.csv", sep=""),
		stringsAsFactors=F,
		strip.white=T,
		na.strings = c("NA", "", "#DIV/0!"),
		sep=",",
		header=T,
		comment.char = "#"
	)
	ds
}

rmMostlyNA <- function(trn) {
	cols <- names(trn)
	pctNA <- data.frame(row.names="pctNA")
	for (col in cols) {
		pctNA[,col] <- sum(is.na(trn[,col])) / nrow(trn)
	}
	trn[,names(pctNA[,pctNA[1,]<0.95,drop=F]), drop=F]
}

# The first thing I did is figure out if I can ignore any of these variables in the training dataset
findNearZero <- function(trn) {
	nzv <- nearZeroVar(trn, saveMetrics=T)
	trn[,names(as.data.frame(t(nzv[nzv$nzv==F,])))]
}

# This didn't work because not all columns are numeric, and i dunno if it's right anyway,
# so why not just use caret's PreProcess
standardize <- function(trn) {
	trn - colMeans(trn) / apply(trn, 2, sd)
}

findNonNum <- function(df) {
	# apply calls as.matrix, which apparently coerces all elements of the data.frame into one type
	# nonnum <- apply(df, 2, class)
	# So use sapply instead
	arow <- df[1,] # Just use one row
	classes <- sapply(arow, class)
	return(names(classes[classes=="character" | classes=="factor"]))

	chars <- names(classes[classes=="character"])
	chars <- chars[! chars %in% c("user_name", "cvtd_timestamp", "new_window", "classe")]
	ndf <- df[,chars]
	for (c in chars) {
		ndf[,c] <- as.numeric(ndf[,c])
	}
	ndf
}

compare <- function(row) {
	print(class(row))
	print(names(row))
	o <- row$o
	n <- row$n
	if (is.na(o) && is.na(n)) return(T)
	else if(as.character(o) == as.character(n)) return(T)
	else return(F)
}

doPCA <- function(df) {
	chars <- findNonNum(df)
	prComp <- prcomp(df[!names(df) %in% chars])
	prComp
}

getPreProcess <- function(df, cols) {
	preObj <- preProcess(df[,cols], method=c("center", "scale", "pca"))
	preObj
}

doAll <- function() {
	training <- readDataset("train")
	testing <- readDataset("test")
	training <- rmMostlyNA(training)
	training <- findNearZero(training)

	cols <- names(training)
	
	testing <- testing[,cols[cols!="classe"]]
	charcols <- findNonNum(training)
	numcols <- cols[! cols %in% charcols]

	pp <- getPreProcess(training, numcols)

	print("Calling predict on training")
	training2 <- predict(pp, training[,numcols])
	print("Calling predict on testing")
	testing2 <- predict(pp, testing[,numcols])

	training2[,"classe"] <- factor(training[,"classe"])
	training2
}

readTrain <- function() {
	training <- readDataset("train")
	training <- rmMostlyNA(training)
	training <- findNearZero(training)
	training
}

doCorrOrig <- function(training, threshold=0.9) {
	training <- rmColumns(training)
	training$classe <- as.numeric(factor(training$classe))
	ccc <- cor(training)
	training[, -findCorrelation(ccc)]
}

doCorr <- function(wl, threshold=0.9) {
	trn <- wl$training
	ccc <- cor(trn[,names(trn)!="classe"])
	correlated <- findCorrelation(ccc)
	wl$training <- trn[, -correlated]

	tstcols <- names(wl$training)
	tstcols <- tstcols[tstcols %in% names(wl$testing)]

	wl$testing <- wl$testing[,tstcols]
	wl
}

readTrain2 <- function() {
	training <- readDataset("train")
	print("removing columns")
	training <- rmColumns(training)
	print("making classe a number")
	training$classe <- as.numeric(factor(training$classe))
	print("running nearZeroVar")
	training <- training[, -nearZeroVar(training)]
	return(training)
	print("finding correlations")
	training <- training[, -findCorrelation(cor(training), 0.95)]
	training
}

readTest <- function(trn) {
	testing <- readDataset("test")
	cols <- names(trn)
	testing <- testing[,cols[cols!="classe"]]
	testing
}

getPP <- function(trn, donotpp) {
	pp <- vector("list", 1)
	cols <- names(trn) %in% donotpp
	pp$centerScale <- preProcess(trn[,!cols], method=c("center", "scale"))
	pp$pca <- preProcess(trn[,!cols], method="pca")
	pp
}

exPPOrig <- function(df, pp, charcols, numcols) {
	df2 <- predict(pp$centerScale, newdata=df[,numcols])
	df2 <- predict(pp$pca, newdata=df2)
	for (col in charcols) {
		df2[,col] <- factor(df[,col])
	}
	df2$X <- df$X
	df2
}

exPP <- function(df, pp, donotpp) {
	cols <- names(df) %in% donotpp
	df2 <- predict(pp$centerScale, newdata=df[,!cols])
	df2 <- predict(pp$pca, newdata=df2)
	df2
}

getFolds <- function(trn) {
	createFolds(trn$classe)
}

rmColumns <- function(df) {
	cols <- names(df)
	ignore_cols <- c("user_name", "cvtd_timestamp", "raw_timestamp_part_1", "raw_timestamp_part_2", "num_window")
	keep_cols <- cols[!cols%in%ignore_cols]
	df[,keep_cols]
}

qplotAll <- function(df) {
	par(ask=T)
	for (column in names(df)) {
		if (! column %in% c("X", "classe")) {
			print(qplot(1:dim(df)[1], df[,column], col=df$classe, main=column))
		}
	}
	par(ask=F)
}

lift <- function() {
	set.seed(12453)

	print("Read datasets")

	training <- readTrain()
	testing <- readTest(training)

	print("Remove some columns")

	training <- rmColumns(training)
	testing <- rmColumns(testing)

	print("Setup wl")

	wl <- list()
	wl$training <- training
	wl$testing <- testing
	wl$classe <- training$classe
	wl$classeF <- factor(training$classe)

	print("Remove correlated columns")

	wl <- doCorr(wl)

	cols <- names(wl$training)
	charcols <- findNonNum(wl$training)
	numcols <- cols[! cols %in% charcols]

	print("Checking for cols in testing not in training")
	print(names(testing)[! names(testing) %in% names(training)])

	print("Get PP")

	donotpp <- c("classe", "X")
	wl$pp <- getPP(wl$training, donotpp)

	print("Preprocess training")

	wl$pptraining <- exPP(wl$training, wl$pp, donotpp)
	wl$pptraining$classe <- wl$training$classe

	print("Preprocess testing")

	wl$pptesting <- exPP(wl$testing, wl$pp, donotpp)

	wl
}

scorePreds <- function(vals, preds) {
	sum(preds==vals) / length(vals)
}

getRpart <- function(data, folds) {
	foldix = 1
	training <- data[-folds[[foldix]],]
	validation <- data[folds[[foldix]],]
	m <- train(classe ~ ., data=training, method="rpart")
	print(m)
	res <- predict(m, newdata=validation)
	print(missClass(validation$classe, res))
	m
}

getRF <- function(trn) {
	folds <- createFolds(trn$classe)
	res <- vector("list", length(folds))
	res$folds <- folds
	res$foldix = 4
	trnfold <- trn[-folds[[res$foldix]],]
	valfold <- trn[folds[[res$foldix]],]
	res$rfFit <- train(classe ~ ., data=trnfold, method="rf")
	res$preds <- predict(res$rfFit, newdata=valfold)
	res$score <- scorePreds(valfold$classe, res$preds)
	res
}

getMultiLogit <- function(data, folds) {
	foldx <- 3
	training <- data[-folds[[foldx]],]
	validation <- data[folds[[foldx]],]
	m <- train(classe ~ ., data=training, method="glm", family="multinomial")
	m
}

getRF2 <- function(data, folds) {
	foldix = 3
	training <- data[-folds[[foldix]],]
	validation <- data[folds[[foldix]],]
	m <- train(classe ~ PC9 + PC10 + PC12 + PC13 + PC15, data=training, method="rf")
	print(m)
	res <- predict(m, newdata=validation)
	print(missClass(validation$classe, res))
	m
}

bagIt <- function(trn) {
	predictors <- trn[!names(trn)=="classe"]
	outcome <- trn$classe
	treebag <- bag(predictors, outcome, B=10,
		bagControl = bagControl(
			fit = ctreeBag$fit,
			predict = ctreeBag$pred,
			aggregate = ctreeBag$aggregate
		)
	)
	treebag
}

foldLM <- function(trn) {
	trn$classe <- as.numeric(trn$classe)
	folds <- createFolds(trn$classe)
	res <- vector("list", length(folds))
	for(i in 1:length(folds)) {
		print(paste0("Building model #", i))
		lmFit <- train(classe ~ ., data=trn[-folds[[i]],], method="lm")
		preds <- predict(lmFit, newdata=trn[folds[[i]],])
		score <- scorePreds(as.numeric(factor(trn[folds[[i]],]$classe)), round(preds))
		res[[i]]$lmFit <- lmFit
		res[[i]]$preds <- preds
		res[[i]]$score <- score
		print(paste("Score:", score))
	}
	res
}

foldRpart <- function(trn) {
	folds <- createFolds(trn$classe)
	res <- vector("list", length(folds))
	for(i in 1:length(folds)) {
		lmFit <- train(classe ~ ., data=trn[-folds[[i]],], method="rpart")
		preds <- predict(lmFit, newdata=trn[folds[[i]],])
		score <- scorePreds(as.numeric(factor(trn[folds[[i]],]$classe)), round(preds))
		res[[i]]$lmFit <- lmFit
		res[[i]]$preds <- preds
		res[[i]]$score <- score
	}
	res
}

foldRF <- function(trn) {
	folds <- createFolds(trn$classe)
	res <- vector("list", length(folds))
	for(i in 1:length(folds)) {
		print(paste0("Building random forest #", i))
		rfFit <- randomForest(classe ~ ., data=trn[-folds[[i]],])
		preds <- predict(rfFit, newdata=trn[folds[[i]],])
		score <- scorePreds(as.numeric(factor(trn[folds[[i]],]$classe)), round(preds))
		res[[i]]$rfFit <- rfFit
		res[[i]]$preds <- preds
		res[[i]]$score <- score
		print(paste("Score:", score))
	}
	res
}

foldNB <- function(trn) {
	folds <- createFolds(trn$classe)
	res <- vector("list", length(folds))
	for(i in 1:length(folds)) {
		print(paste0("Building model #", i))
		trnfold <- trn[-folds[[i]],]
		valfold <- trn[folds[[i]],]
		nbFit <- naiveBayes(trnfold[,names(trnfold)!="classe"], trnfold[,"classe"])
		preds <- predict(nbFit, newdata=valfold)
		score <- scorePreds(valfold$classe, preds)
		res[[i]]$nbFit <- nbFit
		res[[i]]$preds <- preds
		res[[i]]$score <- score
		print(paste("Score:", score))
	}
	res
}

buildAndValidate <- function() {
	wl <- lift()
	res <- getRF(wl$pptraining)
	res
}

