## This script is an adapted version of the one originally created for
## for LASI'19 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi19/lasi19-workshops-tutorials/)


## The script provides an example of the overall process of text classification, 
## including: 
## - preprocessing of textual data;
## - transformation of unstructured (textual) data into a structured data format that
##   can be fed into a classification algorithm; this includes feature weighting and 
##   selection, as well as methods for reducing / transforming the feature space,
##   that is, turning a large number of sparse features into a significantly smaller 
##   number of dense features;
## - application of classification algorithms on the transformed textual data 
##   (that is, the created feature set);
## - evaluation of the classification results.
##
## The example is based on a subset of the 20 Newsgroups dataset:
## https://www.kaggle.com/crawford/20-newsgroups

## Even though the 20 Newsgroup dataset allows for multiclass classification,
## to make this example easier to follow and understand, we will limit ourselves 
## to a binary classification task. Note that the same procedure is applicable  
## to multiclass classification, only the computation process would be more 
## demanding and the evaluation measures would be computed somewhat differently. 

## For the binary classification task, we will use data for two groups within 
## the 'politics' topic: the group discussing guns ('talk.politics.guns') and
## the one on Mideast ('talk.politics.mideast'). 
## Being topically closely related, the two groups will pose a challenge for a 
## classifier since it is not an easy task to differentiate between groups of posts 
## where topics are closely related and overlaps in the vocabulary are to be 
## expected (at least far more than between posts on, e.g., space and medicine 
## that were the subject of other newsgroups in the 20 Newsgroups dataset).     


##############################################
## SET UP the SESSION: 
## - INSTALL and LOAD the REQUIRED LIBRARIES
## - LOAD the AUXILIARY SCRIPTS
##############################################

# The script makes use of the following R packages:
# - caret, e1071 - for various ML tasks
# - rpart - for building a decision tree classifier
# - ranger - for building a Random Forest classifier
# - pROC - for computing ROC-related evaluation measures 
# - quanteda - for various text analytics tasks
# - stringr - for advanced string processing
# - irlba - for singular vector decomposition (SVD)
# - dplyr, tidyr - for general data analysis tasks
# - ggplot2 - for visualization
# - janitor - for some data cleaning tasks
# - doSNOW - for multi-core parallel process execution 
# If you miss any of these packages, install them, before proceeding with the script
# install.packages(c("<package_name_1>", "<package_name_2>", ...))

# Initially, we will load just a basic set of R packages 
# whereas the others will be loaded along the way, as we need them
library(dplyr)
library(readr)
library(tidyr)

# Load a set of auxiliary functions
source("tm_utils.R")

# Set the seed to be used in various computations that depend on random processes.
# This is to assure reproducibility of the results.
seed <- 8220


##########################################################
## LOAD CLEANED DATA from csv files and
## SELECT TWO NEWSGROUPS for a BINARY CLASSIFICATION TASK
##########################################################

# We’ll start by reading in the 'cleaned' newsgroup dataset, that is, by 
# reading the CSV files that resulted from preprocessing the original dataset
# through the steps implemented in the preprocess_20News_dataset.R script. 
# The 'cleaned' data were obtained by removing some 'extra' text that 
# we will not need for this analysis, including: 
# - post header
# - automated email signatures 
# - lines of text refering to the quotations from other users
# NB.1: since this 'extra' text has been removed using some simple heuristics,
# it might happen that not all these extra text bits were removed. 
# NB.2: data stored in the header of a post may be useful for
# classification purposes. However, in this case, we will restrict
# our analysis to the content of the discussion.

# Load training data
train_posts <- read_csv("data/20news-bydate-train.csv") %>% 
  data.frame()
glimpse(train_posts)

# Load test data
test_posts <- read_csv("data/20news-bydate-test.csv") %>%
  data.frame()
glimpse(test_posts)


# Keep only the data related to the two chosen newsgroups
unique(train_posts$newsgroup)

selected_ngs <- c('talk.politics.guns', 'talk.politics.mideast') 
selected_lbls <- c('guns', 'mideast')

train_2cl <- train_posts %>%
  filter(newsgroup %in% selected_ngs) %>%
  mutate(newsgroup = factor(newsgroup, # transform newsgroup into a factor w/ shorter labels
                            levels = selected_ngs,
                            labels = selected_lbls))
summary(train_2cl)

test_2cl <- test_posts %>%
  filter(newsgroup %in% selected_ngs) %>%
  mutate(newsgroup = factor(newsgroup, 
                            levels = selected_ngs,
                            labels = selected_lbls))
summary(test_2cl)

# We will now use the training set to build a classifier.
# Test set will be used later, only for evaluation purposes.

####################################
## DATA (TEXT) WRANGLING: 
## TEXT CLEANING AND TRANSFORMATION 
####################################

# There are many packages in the R ecosystem for performing text analytics.
# One of the latest is *quanteda*. It has many useful functions for quickly
# and easily working with text data; they are well explained in the
# quanteda docs: https://quanteda.io/
library(quanteda)

#
# Create a text corpus
#

# Create a quanteda corpus object out of the posts' content
train_corpus <- corpus(train_2cl$post_txt)

# Add newsgroup as a document level variable, to represent, for each
# post, the class the post belongs to
docvars(train_corpus, field = "newsgroup") <- train_2cl$newsgroup

# By having the corpus object, we'll be able to simplify some later
# processing steps; in addition, we can easily compute some basic
# corpus statistics.

# Get a summary for the first 10 documents in the corpus:
summary(train_corpus, n = 10)


#
# Tokenization of posts
#

# When tokenizing documents, a typical practice is to remove punctuation characters
# and symbols; since these are defined in quanteda via Unicode categories, you may find 
# this table useful: https://unicode.org/reports/tr44/#GC_Values_Table
# Numbers and urls are also often removed. However, we'll keep numbers and urls for now
# and deal with them later. 
?tokens
train_tokens <- tokens(x = train_corpus, 
                       what = "word", 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE,
                       remove_separators = TRUE)

# Take a look at a specific post and see how it has been transformed
train_2cl$post_txt[15]
train_tokens[[15]]

# Next, we will reduce all tokens to lower letters to reduce the variability of 
# the token set (a part of the process known as text normalization)
train_tokens <- tokens_tolower(train_tokens)
train_tokens[[15]]

# Replace numbers (if present) with the "NUMBER" entity.
# The rationale: we are not interested in specific numbers, just the fact that a
# number appeared in a post
train_tokens <- tokens_replace(x = train_tokens, 
                               pattern =  "^[-\\+]?\\d*[,\\.]?\\d*$", 
                               replacement = "NUMBER",
                               valuetype = 'regex',
                               verbose = TRUE)
train_tokens[[15]]


# Next, replace urls (if present) with the "URL" entity
# The same kind of rationale as for numbers.
# First check if they are present
url_pattern <- "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
train_tokens <- tokens_replace(x = train_tokens,
                            pattern = url_pattern, 
                            replacement = "URL",
                            valuetype = "regex",
                            verbose = TRUE) 

# NB: the used URL regex pattern is taken from: http://urlregex.com/
# It is not a perfect one, but covers majority of cases.


## Note: regular expressions are very handy and often indispensable for text cleaning and
## transformation. If you feel you need to learn about regex or refresh your memory, 
## this tutorial is excellent: http://regex.bastardsbook.com/
## Also, the following R cheatsheet comes in useful:
## https://www.rstudio.com/wp-content/uploads/2016/09/RegExCheatsheet.pdf 


# Looking again at the example post, one may observe the presence of tokens with 1 or 2 
# characters only; these will be removed as they rarely bear any relevance for topical
# classification of documents
train_tokens <- tokens_keep(x = train_tokens, min_nchar = 3)
train_tokens[[15]]

# Note: you may also consider removing overly long tokens, but as these would be
# rather rare (ie. have very low frequency), they would be, anyway, removed in the
# feature selection/filtering phase.


# Since forum posts, as well as messages exchanged in other kinds of online  
# communication channels (e.g. chat, status posts), tend to have misspelled 
# words, it might be useful to do spelling correction, as a part of the text 
# normalization step. A typical approach is to check the text against some of 
# the available misspelling corpora - see this example:
# https://cran.r-project.org/web/packages/fuzzyjoin/vignettes/stringdist_join.html
# There is also an R package - spelling - for spell checking:
# https://github.com/ropensci/spelling
# We will skip this step for now.


# Next, we will remove stopwords.
# To that end, we will use quanteda's default stopwords list for English.
# https://github.com/quanteda/stopwords

# It is advised to inspect the default stopword list before
# applying it to the problem at hand - the default one may not be suitable
# for the task at hand (e.g. for sentiment / emotion detection personal  
# pronouns are typically preserved).
head(stopwords(), n = 50)
tail(stopwords(), n = 50)
train_tokens <- tokens_remove(train_tokens, stopwords())
train_tokens[[15]]

# Note: depending on the task at hand, you might want to extend the 
# built-in stopword list with additional, corpus specific 'stopwords'
# (e.g. overly frequent words in the given corpus).


# Perform stemming on the tokens
# (the function uses Porter's stemming algorithm) 
train_tokens <- tokens_wordstem(train_tokens, language = "english")
train_tokens[[15]]


# In case you need lemmatisation for the task at hand,
# consider using the *udpipe* R package:
# https://github.com/bnosac/udpipe
# as it offers language models for a number of languages 
# and using such models, you can lemmatise words

###################################
# CREATE DOCUMENT TERM MATRIX (DTM)
###################################

# Now, we are ready to create DTM. 
# In quanteda's terminology DTM is referred to as "document feature matrix" or dfm
?dfm
train_dfm <- dfm(x = train_tokens, 
                 tolower = FALSE)  # we've already lower cased the tokens

train_dfm
# It's very sparse (sparsity = the proportion of cells that have zero counts); 

# By default, words are weighted with term frequencies (TF)
View(as.matrix(train_dfm)[1:20,1:20])

# Given the large number of features (~16.6K) and the high level of
# sparsity, we should do feature selection. 

####################
# FEATURE SELECTION
####################

# We will use the Chi2 test of independence to do feature selection.
# The null hypothesis (H0) of the test is that the occurrence of a term 
# and the occurrence of a class label are independent; so, if a test
# proves to be significant, we reject the null hypothesis and 
# consider some dependence between the term and the class label.

# Since Chi2 is not reliable for low frequency terms,
# first, we will remove such terms and then compute Chi2

# Examine total (corpus-level) frequency for each token (word)
summary(colSums(train_dfm))
# Summary stats suggest very uneven distribution with at least 
# half of the words appearing only once in the overall corpus.

# So, as the initial selection step, we will remove tokens (words) 
# with only one occurrence in the whole corpus. 
tf_total <- colSums(train_dfm)
to_keep <- which(tf_total > 1)
train_dfm_reduced <- dfm_keep(train_dfm, 
                              pattern = names(to_keep),
                              valuetype = "fixed", 
                              verbose = TRUE)
train_dfm_reduced
# a significant reduction: from ~16.6K to ~8.3K features

# Next, we will use chi2 to select the most discriminating features.
chi2_vals <- dfm_group(train_dfm_reduced, "newsgroup") %>%
  textstat_keyness(measure = "chi2")
head(chi2_vals)
tail(chi2_vals)
# we're not interested in the sign, only in the strength of 
# the association; so, we'll take the absolute value of chi2
chi2_vals <- chi2_vals %>%
  mutate(chi2 = abs(chi2)) %>%
  arrange(desc(chi2))
head(chi2_vals, n=10)
tail(chi2_vals, n=10)

# To determine how to reduce the number of features based on the
# computed chi2 values, consider, the number of features that 
# are significantly associated with the class labels at alpha=0.05
chi2_vals %>%
  filter(p < 0.05) %>%
  nrow()
# 2106 words, which looks fine

# So, we'll keep only the features with statistically significant association 
# with the class labels at alpha=0.05
train_dfm_chi2 <- dfm_keep(train_dfm_reduced, 
                           pattern = chi2_vals$feature[chi2_vals$p < 0.05],
                           valuetype = "fixed", verbose = TRUE)
train_dfm_chi2


# Next, we use the (reduced) DTM to setup a feature data frame 
# with (class) labels. It will serve as the input for a 
# classification algorithm. 
# To create such data frame, we need to do the following:
# - transform quanteda's dfm to a 'regular' R data.frame
# - check for (and remove) documents (rows) that have 'lost' 
#   all their words in the feature selection step
# - assure that feature (column) names are regular R names
# This is done by the create_feature_df() f., defined in the 
# tm_utils.R script
train_df <- create_feature_df(train_dfm = train_dfm_chi2, 
                              class_labels = train_2cl$newsgroup)


########################################################
# BUILD the 1st ML MODEL: 
# RPART + UNIGRAMS + TF weight + CHI2 FEATURE SELECTION
########################################################

# As per best practices, we will leverage cross validation (CV) for our
# modeling process. In particular, we will perform 5-fold CV to
# tune parameters and find the best performing model.
# (NB. we restrict ourselves here to 5-fold CV so that the training 
# does not last overly long. When working on your own, better use 
# higher number of folds, typically 10 folds, and even do repeated CV
# to get a better estimation of the model performance).

# The *caret* package will be used for model building through CV
library(caret)

# Note that our data set is not trivial in size. As such, depending on the
# chosen ML algorithm, CV runs might take a long time to complete. 
# To cut down on the total execution time, we use the *doSNOW* R package 
# to allow for CV to run in parallel on multiple (logical) cores.
# Parallel processing, for model building and CV, is directly supported 
# in caret: https://topepo.github.io/caret/parallel-processing.html

# Due to the size of the DTM, at this point, we will use a single decision
# tree (DT) algorithm to build our first model. We will use more powerful algorithms 
# later when we perform feature transformation that will significantly reduce 
# the size of our feature set.

# Load also the *rpart* R package required for building DTs 
library(rpart)

# We will tune the cp parameter, which is considered the most important in the
# rpart function (the function used in the rpart package for building a DT).
# cp stands for the complexity parameter; any split that does not improve the overall
# fit of the model by at least cp is not attempted; default value is 0.01.

# Define the grid of values for the cp parameter to be examined during the CV process
cp_Grid = expand.grid( .cp = seq(from = 0.001, to = 0.02, by = 0.001)) 

# Build a DT classifier through CV 
# (the cross_validate_classifier() function is defined in the tm_utils.R script)
rpart_cv_1 <- cross_validate_classifier(seed,
                                        nclust = 5, # 7 would be the best value, according to caret's doc.
                                                    # however, not possible on my laptop
                                        train_data = train_df,
                                        ml_method = "rpart",
                                        grid_spec = cp_Grid)

# Check out the results:
rpart_cv_1
plot(rpart_cv_1)

# First, take the cp value of the best performing model in CV
m1_best_cp <- rpart_cv_1$bestTune$cp
# Then, extract performance measures for the best cp value 
m1_best_results <- rpart_cv_1$results %>% 
  filter(cp==m1_best_cp)
m1_best_results

## Remainder:
## - sensistivity (true positive rate) = TP/(TP+FN)
##   (e.g., if positive class is passing an exam, sensitivity represents the proportion 
##    of students who passed the exam and were correctly predicted to pass among all 
##    the students who passed the exam)
## - specificity (true negative rate) = TN/(TN+FP)
##    (to continue with the example, specificity would be the proportion of students who 
##    failed the exam and were correctly predicted to fail, among all the students who
##    really failed) 

## Nice explanation of Sensitivity, Specificity, ROC curves and AUC:
## https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5

## Note that in this example we do not have 'true' positive and negative classes,
## that is, no preference for better predicting one class ('guns') over the other 
## ('mideast'). So, a balanced, good performance based on both sensitivity and 
## specificity is what we will be striving for. In other situations, this may not 
## be the case, and you'll put more focus on either sensitivity (focus on the 
## positive class) or specificity (focus on the negative class).
## For example, in case of spam detection, where 'spam' is the 'positive' class,
## one would put focus on maximizing specificity, that is, reducing the number of
## non-spam (aka 'ham') messages that were incorrectly classified as being spam.

# We can use the final model to examine feature importance
rpart_cv_1$finalModel$variable.importance %>%
  sort(decreasing = TRUE) %>%
#  names() %>%
  head(20)

# Another way to inspect relevant features
varImp(rpart_cv_1, useModel = FALSE) %>% plot(top=20)

# Note that these two ways of obtaining feature importance gave somewhat different results.
# The reason is that they apply different calculations to determine feature importance:
# - in rpart: the feature's overall contribution to the decrease in (node) impurity 
# - in caret: the overall reduction in the loss function attributed to a feature
# (for details check the documentation)

# In the next step, we will apply a more sophisticated feature reduction method.
# In particular, we'll apply Singular Value Decomposition (SVD) to the DTM of
# TF-IDF weighted unigrams and bigrams.


#########################################################
# USE UNIGRAMS AND BIGRAMS AS THE INITIAL FEATURE SET;
# APPLY SINGULAR VALUE DECOMPOSITION (SVD) TO TRANSFORM
# THE FEATURE SPACE
#########################################################

# N-grams allow us to augment our DTM matrix with word ordering.
# This tends to lead to increased performance over ML models 
# trained with unigrams only. On the down side, the inclusion of
# ngrams, even only bigrams and trigrams, leads to an explosion 
# in the number of features. So, in addition to unigrams, 
# we will use only bigrams. 


# Starting from the corpus, create a dfm in a similar way 
# it was done before. Since the NUMBER feature did not prove 
# particularly important (see above the list of important features), 
# we can omit it, and simply remove numbers
train_dfm_2 <- train_corpus %>% 
  dfm(tolower = TRUE, 
      remove_punct = TRUE, 
      remove_symbols = TRUE,
      remove_numbers = TRUE,
      remove = stopwords("english"),
      stem = TRUE,
      ngrams = 1:2) %>%
  dfm_keep(min_nchar = 3)
train_dfm_2
# about 148K features

# Considering the number of features, we need to do
# an initial feature reduction before proceding to SVD.

# Compute the overall (corpus) TF for each term
dfm_2_tot_tf <- colSums(train_dfm_2)
summary(dfm_2_tot_tf)
# Considering the (huge) number of terms and very low frequency of a large
# majority of them, keep only those with the overall (corpus) frequency 
# above the 75th percentile
to_keep <- which(dfm_2_tot_tf > quantile(dfm_2_tot_tf, probs = 0.75))
train_dfm_2 <- dfm_keep(train_dfm_2, 
                        pattern = names(dfm_2_tot_tf[to_keep]),
                        valuetype = "fixed", verbose = TRUE)
train_dfm_2
# reduced to ~34K features

# We need to change the weighting scheme, that is, to use
# TF-IDF weights instead of TF. This is required for later
# use of SVD for feature transformation. Besides, TF-IDF is
# consdered a better reflection of terms' relevance than 
# TF metric
train_dfm_2 <- dfm_tfidf(train_dfm_2, 
                         scheme_tf = 'prop', 
                         scheme_df = 'inverse')


# We can now do SVD transformation to reduce the number 
# of features (ngrams) to a significantly smaller set 
# that explains a large portion of variability in the data.

# Suggested reading for SVD and its use in text analysis 
# (Latent Semantic Analysis):
# - Landauer, T. K., Foltz, P. W., & Laham, D. (1998). 
#   Introduction to Latent 
#   Semantic Analysis. Discourse Processes, 25, 259-284. 
#   URL: http://lsa.colorado.edu/papers/dp1.LSAintro.pdf


# To apply SVD to our data, we need to set the number of the most important 
# singular vectors we wish to calculate and retain as features 
# (in SVD terms, it is the rank the original matrix is to be reduced to).
# How to determine the "right" number of singular vectors is still an open issue.
# Some useful links on that topic:
# - https://stackoverflow.com/questions/9582291/how-do-we-decide-the-number-of-dimensions-for-latent-semantic-analysis 
# - https://irthoughts.wordpress.com/2008/02/13/lsi-how-many-dimensions-to-keep/

# We will reduce the dimensionality down to 300 columns. This number is chosen as it
# is often recommended (based on the experience in practice).
# (NB. To get the best results, the number of dimensions would have to be  
# experimentally determined, by trying several different values and comparing 
# the performance of the resulting models)

# We'll use the *irlba* R package for SVD computations
# Note: SVD works with TDM (Term Document Matrix), not DTM. 
# TDM is nothing more than transposed DTM.

library(irlba)
set.seed(seed)
svd_res <- irlba(t(as.matrix(train_dfm_2)), # SVD / LSA requires TDM (not DTM) as its input 
                 nv = 300, # the number of dimensions (singular vectors) to estimate
                 maxit = 600) # max iterations is set to be twice larger than nv 

# (n.b. the above function call takes a few minutes to execute)

# Examine the result:
str(svd_res)
# d - stores the singular values of the original matrix (values on the diagonal of the sigma matrix)
# u - its columns are referred to as left singular vectors of the original matrix; it represents the 
#     relation between the extracted dimensions (in the columns) and the ngrams (in the rows)
# v - its columns are referred to as right singular vectors of the original matrix; it respresents 
#     the relation between the extracted dimensions (columns) and the documents (rows)

# Store these vectors and matrices so that the computation 
# does not have to be repeated
saveRDS(svd_res$d, "models/svd/sigma.RData")
saveRDS(svd_res$u, "models/svd/left_sv.RData")
saveRDS(svd_res$v, "models/svd/right_sv.RData")

# Take a glimpse at the new feature set (the right singular vector):
View(svd_res$v[1:20,1:50])

# Create a new feature data frame using the 300 features obtained by applying
# SVD to TF-IDF weighted DTM (i.e. the V matrix produced by SVD)
train_svd_df <- cbind(Label = train_2cl$newsgroup, 
                      data.frame(svd_res$v))

# Next, we will examine the predictive power of the model with singular 
# vectors as features.

###############################################
# BUILD the 2nd ML MODEL: RANDOM FOREST + 
# SINGULAR VECTORS (FROM TF-IDF WEIGHTED DTM)
###############################################

# We have significantly reduced the dimensionality of our data using SVD. 
# Now, we can use a more complex and powerful classification algorithm. 
# In particular, we will build a Random Forest (RF) model.

## For a brief introduction to the Random Forest algorithm, 
## see the "Bagging and Random Forest" slides (made available as part of the WS materials).
## For more details and an excellent explanation of Random Forest and related algorithms,
## see chapter 8.2 of the Introduction to Statistical Learning book
## http://www-bcf.usc.edu/~gareth/ISL/ 

# We will build a RF model with 1000 trees. 
# We'll try different values of the mtry parameter to find the value that would
# lead to the best performance. 
# The mtry parameter stands for the number of features randomly sampled as 
# candidates at each split. 
# For the mtry parameter, we will consider 10 different values between the minimum
# (1 feature) and the maximum possible value (all features). 
n_features <- ncol(train_svd_df)-1
param_Grid <- expand.grid( .mtry = seq(from = 1, to = n_features, length.out = 10),
                         .splitrule = "gini", # gini is a measure of node 'purity'
                         .min.node.size = c(2,3)) 

# NOTE: The following code takes a long time to run. Here is why:
# We are performing 5-fold CV. That means we will examine each model configuration 
# 5 times. We have 20 configurations as we are asking caret to try 10 different
# values of the mtry parameter and 2 different values of the min.node.size parameter. 
# In addition, we are asking RF to build 1000 trees. Lastly, when the best values 
# for the parameters are chosen, caret will use them to build the final model using 
# all the training data. So, the number of trees we're building is:
# (5 * 20 * 1000) + 1000 = 1,001,000 trees!

# Build a RF classifier
rf_cv_1 <- cross_validate_classifier(seed, 
                                     nclust = 5,
                                     train_data = train_svd_df,
                                     ml_method = "ranger",
                                     grid_spec = param_Grid)

# (n.b. the above f. call takes about an hour to execute)

# Save the model to have a quick access to it later
# saveRDS(rf_cv_1, "models/rf_cv_1.RData")

# Load the saved model
# rf_cv_1 <- readRDS("models/rf_cv_1.RData")

# Check out the results
rf_cv_1
plot(rf_cv_1)

# Extract evaluation measures for the best performing model 
svd_best_mtry <- rf_cv_1$bestTune$mtry
svd_best_min_nsize <- rf_cv_1$bestTune$min.node.size
m2_best_res <- rf_cv_1$results %>% 
  filter(mtry==svd_best_mtry & 
           min.node.size==svd_best_min_nsize)
m2_best_res

# Compare the results with the previously CV-ed models
comparison <- data.frame(rbind(m1_best_results %>% select(-cp), # exclude the cp parameter 
                               m2_best_res %>% select(-c(mtry:min.node.size))), # exclude the 3 model parameters 
                         row.names = c("RPART_TF_CHI2", "RF_SVD"))
# Add a column with the number of features
comparison$NFeatures <- c(ncol(train_df),
                          ncol(train_svd_df))
comparison
# The combined use of the new feature set and a more powerful algorithm significantly 
# improved the results, including the reduction in the variability of the results 
# (see SD values). In addition, the number of features is 10 - 12 times smaller than 
# in the other model; this is highly important as it makes the model less prone to
# overfitting.

# The downside is that the features are not interpretable - we can check for the most 
# important features, as we did before, but that will not be informative as the SVD 
# dimensions are rather obscure
plot(varImp(rf_cv_1), top = 20)


# Before moving to the evaluation of the model, we'll do a bit of cleaning
# to release the memory of the large objects that are no longer needed
remove(rpart_cv_1, tf_total, dfm_2_tot_tf, to_keep, train_tokens, 
       train_dfm, train_dfm_reduced, train_dfm_chi2, train_corpus)

##################
# TEST THE MODEL
##################

# Now, it is time to verify the (better) model (SVD + RF)  
# using the test data we set aside at the beginning of the script.  
# The first stage of the evaluation process is running the 
# test data through our text transformation pipeline of:
# - Tokenization (including removal of punctuation, symbols, 
#   and numbers)
# - Removing tokens less than 3 characters long
# - Lower casing
# - Stopword removal
# - Stemming
# - Adding bigrams
# - Creating DTM and ensuring the test DTM has 
#   the same features (ngrams) as the train DTM
# - Computing TF-IDF weights 
# - Feature set transformation / reduction using SVD  

test_tokens <- tokens(x = test_2cl$post_txt, 
                       what = "word", 
                       remove_punct = TRUE, 
                       remove_symbols = TRUE,
                       remove_numbers = TRUE)

test_tokens <- tokens_keep(x = test_tokens, min_nchar = 3) %>%
  tokens_tolower() %>%
  tokens_remove(stopwords()) %>%
  tokens_wordstem(language = "english") %>%
  tokens_ngrams(n = 1:2)
  
test_dfm <- dfm(test_tokens, tolower = FALSE)

# Compare the train and test DTMs
# (note: train DTM is the one that served as the input for 
# creating the features - SVD dimensions - of the best classifier)
train_dfm_2
test_dfm
# The two DTMs differ in the feature set. This is expected as features
# are ngrams from two different sets of posts (training and test).
# However, we have to ensure that the test DTM has the same n-grams 
# (features) as the training DTM.
# The rationale: we need to represent any new post in the feature space 
# that our classifier 'is aware of' (otherwise, it will report an error)
# and that is the feature space of the training set.

# Transform test_dfm so that it has the same features as the dfm that  
# was used to build features of our best classifier
test_dfm <- dfm_keep(test_dfm, pattern = train_dfm_2)
test_dfm
# Now, test dfm seems to have the same features as the train dfm.
# Let's check if those are really the same features
setdiff(featnames(test_dfm), featnames(train_dfm_2))
setdiff(featnames(train_dfm_2), featnames(test_dfm))
# No difference -> they are exactly the same.

# The next step is to 'project' the test DTM into the same 
# TF-IDF vector space we built for our training data. 
# This requires the following steps:
# 1 - Normalize term counts in each document (i.e. each row)
# 2 - Perform IDF multiplication using training IDF values
#
# NB. We'll use IDF values computed on the training set, since 
# IDF is always computed on a representative and sufficiently large
# corpus, and in production settings (when the classifier is deployed),
# we won't have sufficiently large number of unclassified posts to use
# for IDF computation. Hence, the training set is used as a representative,
# large corpus; altenatively, we may use some other large corpus from 
# the same domain and of the same writing style (e.g. forum posts)

# Normalize term counts in all test posts
test_tf <- dfm_weight(test_dfm, scheme = "prop")

# Next, compute IDF values for the features using the training set
# (Note: the inverse_doc_freq() f. is defined in the tm_utils.R script)
train_idf <- apply(train_dfm_2, 2, inverse_doc_freq)

# Next, calculate TF-IDF using the computed IDF values
test_tfidf <-  apply(test_tf, 1, function(x) x*train_idf)
dim(test_tfidf)
# Note that documents are given in columns, while ngrams are in rows,
# meaning this is TDM; we will keep it this way as this form will be
# required in the next step

# With the test data projected into the TF-IDF vector space of the training
# data, we can now do the final projection into the training SVD space
# (i.e. apply the SVD matrix factorization).

##############################################
## APPLYING SVD PROJECTION ON A NEW DATA SET
##############################################

# The formula for projecting a particular document (d) to the SVD space: 
#
# d_hat = sigma_inverse * transposed_U_matrix %*% d_TF-IDF_vector
#
# d_hat is the representation of the given document d in the SVD space of 
# the training dataset; more precisely, it is the representation of d in
# terms of the dimensions of the V matrix (right singular vectors).
# 
# Before applying this formula, let's first demonstrate that it will really 
# allow us to represent documents using the SVD dimensions. 
# To that end, we'll examine its use on a document from the training set,
# since for documents from the training set we know their SVD representations 
# (they are given in the V matrix)
example_doc <- as.matrix(train_dfm_2)[1,]

# For convenience, we'll introduce:
sigma_inverse <- 1 / svd_res$d # 1 / readRDS("models/svd/sigma.Rdata") 
u_transpose <- t(svd_res$u) # readRDS("models/svd/left_sv.Rdata") %>% t() 

# The projection of the example document in the SVD space, based on the 
# above given formula:
example_doc_hat <- as.vector(sigma_inverse * u_transpose %*% example_doc)
# We expect that the obtained vector is very similar, if not completely equal,
# to the vector (row) in the V matrix which corresponds to the example document.

# Look at the first 10 components of projected document...
example_doc_hat[1:10]
# ... and the corresponding row in the document space produced by SVD (the V matrix)
svd_res$v[1, 1:10]
# v <- readRDS("models/svd/right_sv.Rdata")
# v[1,1:10]
# The two vectors are almost identical (note the values are expressed in e-04, e-05,...).
# In fact, the differences are so tiny that when we compute cosine similarity 
# between the two vectors, the similarity turns to be equal to 1:
library(lsa)
cosine(example_doc_hat, svd_res$v[1,]) # v[1,]
#
# Why is this useful?
# It shows that using the above given formula, we can transform any document into
# the singular vector space of the training set, using the computed sigma_inverse 
# and transposed_U_matrix; this further means that we can take a new, unseen 
# document (a post in our case), represent it as a TF-IDF weighted vector, and 
# transform it into the singular vector space so that it can be classified by our 
# prediction model.


# So, we will use the above given formula to represent posts from the test set in 
# the singular vector space. As we have multiple documents, we need to replace 
# d_TF-IDF_vector (3rd element on the left), with a matrix of TF-IDF values 
# (the matrix should have terms in rows and documents in columns)
test_svd_hat <- sigma_inverse * u_transpose %*% test_tfidf
dim(test_svd_hat)

###################################################
## MAKE PREDICTIONS ON THE (TRANSFORMED) TEST DATA
###################################################

# With the feature set ready, we can now build the test data frame to 
# feed into our prediction model
test_svd_df <- data.frame(Label = test_2cl$newsgroup, 
                          t(test_svd_hat)) # need to transpose it, to place documents in rows

# Now we can make predictions on the test data set 
# using our best classifer (rf_cv_1)
# rf_cv_1 <- readRDS("models/rf_cv_1.Rdata")
preds <- predict(rf_cv_1, newdata = test_svd_df)

# Examine the results
# 1) inspect confusion matrix
cm <- table(Actual = test_svd_df$Label, Predicted = preds)
cm

# 2) compute evaluation measures
# (note that the get_eval_measures() function is defined in tm_utils.R)
eval_metrics <- c('Sensitivity', 'Specificity', 'AUC')
get_eval_measures(rf_cv_1, test_svd_df, eval_metrics)

# Let's compare these results with those obtained on the training data 
m2_best_res[, c('Sens', 'Spec', 'ROC')]

# The performance is lower than on the training set, which is expected
# (one should almost always expect lower performance on the test set)
# However, it is still rather good.

