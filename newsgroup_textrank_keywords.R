## This script is an adapted version of the one originally created for
## for LASI'18 Workshop on Text mining for learning content analysis 
## (https://solaresearch.org/events/lasi/lasi-2018/lasi18-workshops/)


## The script provides an example of applying the TextRank algorithm, 
## using the *textrank* R package, to extract keywords (and key phrases) 
## from science-related newsgroups.
##
## The TextRank algorithm was originally proposed by Mihalcea & Tarau, in:
## Mihalcea, R. & Tarau, P. (2004). TextRank: Bringing order into texts. 
## In D. Lin & D. Wu (Eds.), Proc. of Empirical Methods in Natural Language 
## Processing (EMNLP) 2004 (pp. 404–411), Barcelona, Spain, July. 
## URL: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
##
## The textrank R package is available at:
## https://cran.r-project.org/web/packages/textrank/vignettes/textrank.html
##
## The dataset used in the example is derived from the publicly available
## 20 Newsgroups dataset, http://qwone.com/~jason/20Newsgroups/ 

# Load the initial set of required packages
# (more will be added along the way)
library(tidyr)
library(dplyr)
library(quanteda)
library(stringr)

# Read in a (pre-processed) subset of the 20 Newsgroups dataset
# (for the preprocessing details, see the preprocess_20News_dataset.R script)
usenet_data <- read.csv(file = "data/20news-bydate-train.csv",
                        stringsAsFactors = FALSE)

# Examine the available newsgroups
unique(usenet_data$newsgroup)

# Select only posts originating from science-focused groups
sci_data <- usenet_data %>%
  filter(str_detect(newsgroup, "^sci\\."))
# Remove usenet_data as no longer needed
remove(usenet_data)

# Keywords extraction using the TextRank algorithm requires: 
# - Part-of-Speach (POS) tagged words, and 
# - word lemmas 
# To do both things, we will use the *udpipe* R package
# https://github.com/bnosac/udpipe
# (quanteda package does not support POS tagging, nor lemmatisation)
# install.packages('udpipe')
library(udpipe)

# Load the appropriate language model (the one for English language)
# Note: the udpipe package has language models for over 65 languages
lang_model <- udpipe_download_model("english", model_dir = "models/textrank")
View(lang_model)
tagger <- udpipe_load_model(file = lang_model$file_model)

# Annotate the text of the science posts using the loaded model (tagger).
# Specifically, we need to do tokenisation, POS tagging, and lemmatisation
# (dependency parsing not needed)
sci_posts_annotated <- udpipe_annotate(object = tagger, 
                                       x = sci_data$post_txt,
                                       doc_id = sci_data$id,
                                       tagger = "default",
                                       parser = "none",
                                       trace = TRUE)

# To be able to use the udpipe object easily, we'll transform it 
# into a data frame
sci_posts_ann_df <- as.data.frame(sci_posts_annotated)
# and remove the large udpipe object, to release memory
remove(sci_posts_annotated)

# Store the data frame with annotations since the annotation taks 
# takes a lot of time to complete
saveRDS(sci_posts_ann_df, file = "models/textrank/sci_posts_ann_df.RData")

# Load the saved annotations
# sci_posts_ann_df <- readRDS(file = "models/textrank/sci_posts_ann_df.RData")

# Let's quickly inspect the results of the annotation process
glimpse(sci_posts_ann_df)
View(sci_posts_ann_df[1:100, ])
# The meaning of all the variables is explained at:
# http://universaldependencies.org/format.html

# Next, we'll examine a sample of lemmas to check what they look like 
# and if some text 'cleaning' is required
unique(sci_posts_ann_df$lemma)[1:100]
unique(sci_posts_ann_df$lemma)[2000:2100]
# Note the presence of stopwords ('the', 'and',...), as well as 
# single-char lemmas, numbers, and punctuation marks.

# To clean up the data, we will remove stopwords, numbers, punctuation, ...
sci_posts_ann_filtered <- sci_posts_ann_df %>%
  filter(!lemma %in% stopwords()) %>%
  filter(nchar(lemma) > 1) %>%
  filter(str_detect(lemma, "^[a-z0-9'-]+$")) %>%  # allow only letters, digits, single quotes, and hyphens in lemmas 
  filter(!str_detect(lemma, "^'[a-z]{1,2}$")) %>% # do not allow lemmas consisting of a single quote followed by up to 2 letters (eg. 's, 'll)  
  filter(!upos %in% c('PUNCT', 'NUM'))    # do not allow numbers and punctuation

# Check the results of the 'cleaning' steps
unique(sci_posts_ann_filtered$lemma)[1:100]
unique(sci_posts_ann_filtered$lemma)[2000:2100]

# Note: the above cleaning steps were not really necessary since we will later 
# filter lemmas based on their POS tags, to select only nouns and adjectives,
# but it was a good opportunity to practice text cleaning :-)

# Now that the text is preprocessed, we can proceed to keywords extraction
# using the textrank package
# install.packages('textrank')
library(textrank)

# Check the function for the extraction of keywords
?textrank_keywords

# To extract keywords, we need to provide:
# - a vector of words (lemmas)
# - a vector of logicals indicating which words are potentially relevant, that is, 
#   which words are to be used to build a word collocation graph; we will select 
#   nouns and adjectives
# - the proportion or the number of words to keep
# - the longest n-gram to consider when forming keyphrases

# To determine the proportion of words to keep, we will
# check the number of nouns and adjectives (as relevant word types)
rel_word_types <- c("NOUN", "ADJ")
sci_posts_ann_filtered %>%
  filter(upos %in% rel_word_types) %>%
  select(lemma) %>%
  n_distinct()
# ~12.4K nouns and adjectives
# Considering this large number of potentially relevant words,
# we will tell the function to keep just 2.5% (~300) as relevant

# Run the textrank algorithm
sci_keywords <- 
  textrank::textrank_keywords(x = sci_posts_ann_filtered$lemma,
                              relevant = sci_posts_ann_filtered$upos %in% rel_word_types,
                              p = 0.025,  
                              ngram_max = 3,
                              sep = "_")
# Examine the structure of the result
str(sci_keywords)
# The 1st component of the result list is the 'terms' vector with top ranked words
# based on the computed PageRank score: 
head(sci_keywords$terms, n=15)
# The disadvantage is that we do not have PageRank score associated with these words

# To get PageRank score, we should take the 2nd component of the result (pagerank):
head(sci_keywords$pagerank$vector, n=15)
# Note that this vector contains PageRank score for all the considered words, not 
# only for the proportion we specified to be kept (0.025 in this case)

# Sort the keywords, based on the PageRank
sci_keyw_pagerank <- sort(sci_keywords$pagerank$vector, decreasing = TRUE)
head(sci_keyw_pagerank, n = 20)
# Overall the keywords look fine, though seem to be overly general. This is, in fact, 
# expected considering the broad range of topics dicussed in the science-related newsgroups. 
# Hence, it might be better to focus on individual newsgroups instead.


# To get keywords of individual science newsgroups and then compare them, 
# we need to call the textrank_keywords() f. for each newsgroup separatelly.

# First, since the data frame with annotated documents (sci_posts_ann_filtered) 
# does not have a connection between the documents (posts) and newsgroups, we need 
# to establish that connection:
sci_posts_ann_filtered <- merge(x = sci_posts_ann_filtered,
                                y = sci_data %>% select(id, newsgroup),
                                by.x = "doc_id", by.y = "id", 
                                all.x = TRUE, all.y = FALSE) 

# Next, we'll create a function that for the given set of word lemmas, POS tags,
# and proportion of words to keep as keywords, applies the TextRank algorithm and 
# returns the results (lemmas w/ their PageRank scores and freqs) as a data frame.
# It will be used for extracting keywords for each newsgroup individually
tr_keywords <- function(word_lemmas, pos_tags, prop_to_keep) {
  relevant_word_types <- c("NOUN", "ADJ")
  tr_res <- textrank::textrank_keywords(x = word_lemmas,
                                          relevant = pos_tags %in% relevant_word_types,
                                          p = prop_to_keep,
                                          ngram_max = 3,
                                          sep = "_")
  keyw_pagerank <- sort(tr_res$pagerank$vector, decreasing = TRUE)
  keyw_pagerank_df <- data.frame(keyword = names(keyw_pagerank),
                                 pagerank = as.numeric(keyw_pagerank),
                                 ngram = 1,
                                 stringsAsFactors = FALSE) 
  # keep only those words that are selected by the algorithm as relevant
  keyw_pagerank_df <- keyw_pagerank_df %>% filter(keyword %in% tr_res$terms)
  # add frequencies for the selected keywords
  keyw_pagerank_df <- merge(x = keyw_pagerank_df, 
                            y = tr_res$keywords %>% filter(ngram==1) %>% select(keyword, freq),
                            by = "keyword", all.x = TRUE, all.y = FALSE)
  # add ngrams, n > 1
  key_ngrams_df <- tr_res$keywords %>% filter(ngram > 1) %>% mutate(pagerank = NA) 
  key_ngrams_df <- key_ngrams_df[,colnames(keyw_pagerank_df)]
  keyw_pagerank_df <- rbind(keyw_pagerank_df, key_ngrams_df)
  
  keyw_pagerank_df
}

# Before calling our tr_keywords() function on each newsgroup, we have to decide
# on how many words to keep as keywords. To do that, we'll check the number of 
# candidate keywords, that is, nouns and adjectives per newsgroup
sci_posts_ann_filtered %>% 
  filter(upos %in% rel_word_types) %>%
  group_by(newsgroup) %>% 
  summarise(n_candidates = n_distinct(lemma))
# All newsgroups except one have above 5K words (ranges from 4.2K to 5.6K).
# So, it might be good to keep 5% of the top ranked words in each group.

# Let's try the function on the medical posts
med_posts_data <- sci_posts_ann_filtered %>% filter(newsgroup == "sci.med")
med_kwds_df <- tr_keywords(word_lemmas =  med_posts_data %>% pull(lemma),
                       pos_tags = med_posts_data %>% pull(upos),
                       prop_to_keep = 0.05)

med_kwds_df %>% arrange(desc(pagerank)) %>% head(15)

# To visually inspect the results, plot the top words 
# (based on the TextRank score) 
plot_top_kwds <- function(kwds_df, top_n, fill_color='steelblue') {
  require(ggplot2)
  kwds_df %>% 
    top_n(n = top_n, wt = pagerank) %>%
    mutate(lemma = reorder(keyword, pagerank)) %>%
    ggplot(mapping = aes(x = lemma, y = pagerank)) +
    geom_col(show.legend = FALSE, fill=fill_color) +
    coord_flip() +
    theme_bw()
} 

plot_top_kwds(med_kwds_df, 20)


# Let's see how we can use wordclouds as another way to visually inspect
# keywords from a newsgroup  
# install.packages('wordcloud')
library(wordcloud)

# Word clouds typically work with word frequencies (counts);
# to use them to visualise words based on their TextRank scores (very small floats),
# we need to rescale those scores
set.seed(9220)
wordcloud(words = med_kwds_df$keyword[med_kwds_df$ngram==1], 
          max.words = 100,
          freq = round(med_kwds_df$pagerank[med_kwds_df$ngram==1] * 10000), # instead of freq, we are using rescalled pagerank score
          random.order=FALSE,                             # plot words in decreasing freq order
          rot.per=0.35,                                   # prop. of words plotted vertically
          colors=brewer.pal(9, "Purples")) # color pallet

## To learn more about different options to customise word clouds, and also
## some specific kinds of word clouds (e.g. comparison cloud), check this RPub:
## https://rpubs.com/brandonkopp/creating-word-clouds-in-r
## For the selection of a color pallet, consider: http://colorbrewer2.org/
## and the following R color palettes cheatsheet:
## https://www.nceas.ucsb.edu/~frazier/RSpatialGuides/colorPaletteCheatsheet.pdf


# So far, we have only examined keywords (unigrams), but not key phrases
# (bigrams, trigrams). Note that the keyphrases are built out of the keywords.
# We can directly examine the most frequent bigrams and trigrams 
med_kwds_df %>% 
  filter(ngram == 2) %>% arrange(desc(freq)) %>% select(keyword, freq) %>% top_n(15)

med_kwds_df %>% 
  filter(ngram == 3) %>% arrange(desc(freq)) %>% select(keyword, freq) %>% top_n(15)
# Bigrams seem to be more useful as key phrases than trigrams


# TASK:
# Apply the same procedure to extract and examine keywords and keyphrases from the
# posts of the other 3 science-related newsgroups.