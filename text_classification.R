# Author: David M. Louis
# Contact: dmlouis87@gmail.com

##################################################################################
# Library and Setup
##################################################################################
# Data Wrangling
library(tidyverse)
# Text Preprocessing
library(tidytext)
library(textclean)
library(hunspell)
# Model Evaluation
library(yardstick)
# Naive Bayes
library(e1071)
# Deep Learning
library(keras)
library(tensorflow)
library(ggplot2)

reticulate::conda_list()$name
# install_tensorflow()
tf_config()
reticulate::py_discover_config()
reticulate::use_condaenv("r-tensorflow")
reticulate::py_config()
# ggplot2 Plot Configuration
theme_set(theme_minimal() +
            theme(legend.position = "top")
)


##################################################################################
# data
##################################################################################

df <- readxl::read_excel("data/data.xlsx")


##################################################################################
# text preprocessing
##################################################################################
#text cleansing
cleansing_text <- function(x) x %>% 
  replace_non_ascii() %>% 
  tolower() %>% 
  str_replace_all(pattern = "\\@.*? |\\@.*?[:punct:]", replacement = " ") %>% 
  replace_url() %>% 
  replace_hash() %>% 
  replace_html() %>% 
  replace_contraction() %>% 
  replace_word_elongation() %>% 
  str_replace_all("\\?", " questionmark") %>% 
  str_replace_all("\\!", " exclamationmark") %>% 
  str_replace_all("[:punct:]", " ") %>% 
  str_replace_all("[:digit:]", " ") %>% 
  str_trim() %>% 
  str_squish()

cleansing_text("I really love this!!!")

library(furrr)
plan(multisession, workers = 4) # Using 4 CPU cores

df_clean <- df %>% 
  mutate(
    text_clean = text_field  %>% 
      future_map_chr(cleansing_text)
  ) 

head(df_clean)

#checking word count and length
word_count <- map_dbl(df_clean$text_clean, function(x) str_split(x, " ") %>% 
                        unlist() %>% 
                        length()
)

summary(word_count)

#getting worthwhile text field
# df_cleaned <- df_clean %>%
#   filter(word_count > 1)

df_clean <- df_clean %>% mutate(Follow_Up = ifelse(Follow_Up == "no_follow_up",0,1))
df_clean <- df_clean %>% drop_na(Follow_Up)

glimpse(df_clean)


tidy_df <- df_clean %>%
  unnest_tokens(word, text_clean) %>%
  group_by(word) %>%
  filter(n() > 10) %>%
  ungroup()

tidy_df <- tidy_df %>% mutate(Follow_Up = ifelse(Follow_Up == "0","no_follow_up","follow_up"))

#### observing most frequent words
tidy_df %>%
  count(Follow_Up, word, sort = TRUE) %>%
  anti_join(get_stopwords()) %>%
  group_by(Follow_Up) %>%
  top_n(20) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(word, n, Follow_Up), n,
             fill = Follow_Up
  )) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  scale_x_reordered() +
  coord_flip() +
  facet_wrap(~Follow_Up, scales = "free") +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    x = NULL, y = "Word count",
    title = "Most frequent words after removing stop words",
  )


##################################################################################
# cross-validation
##################################################################################
#split the data into train and test
set.seed(123)
row_data <- nrow(df_clean)
index <- sample(row_data, row_data*0.8)

data_train <- df_clean[ index, ]
data_test <- df_clean[-index, ]

#testing class imbalance
table(data_train$Follow_Up) %>% 
  prop.table()

barplot(table(data_train$Follow_Up) %>% 
       prop.table(), names.arg=c("No Follow Up", "Follow Up"))


##################################################################################
# deep learning with lstm
##################################################################################
##### tokenization
paste(data_train$text_clean, collapse = " ") %>% 
  str_split(" ") %>% 
  unlist() %>% 
  n_distinct()

#determing number of words to use
num_words <- 560
tokenizer <- text_tokenizer(num_words = num_words) %>% 
  fit_text_tokenizer(data_train$text_clean)
# Maximum Length of Word to use
maxlen <- 250

##### padding text sequence
#keeping lenght of text the same. Limiting it to 200 words
train_x <- texts_to_sequences(tokenizer, data_train$text_clean) %>% 
  pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")
test_x <- texts_to_sequences(tokenizer, data_test$text_clean) %>% 
  pad_sequences(maxlen = maxlen, padding = "pre", truncating = "post")
# Transform the target variable on data train
train_y <- data_train$Follow_Up
dim(train_x)

##### model architecture
# Set Random Seed for Initial Weight
tensorflow::tf$random$set_seed(123)
# Build model architecture
model <- keras_model_sequential(name = "lstm_model") %>% 
  layer_embedding(name = "input",
                  input_dim = num_words,
                  input_length = maxlen,
                  output_dim = 8
  ) %>% 
  layer_lstm(name = "LSTM",
             units = 8,
             kernel_regularizer = regularizer_l1_l2(l1 = 0.05, l2 = 0.05),
             return_sequences = F
  ) %>% 
  layer_dense(name = "Output",
              units = 1,
              activation = "sigmoid"
  )
model

##### model fitting
model %>% 
  compile(optimizer = optimizer_adam(lr = 0.001),
          metrics = "accuracy",
          loss = "binary_crossentropy"
  )
epochs <- 9
batch_size <- 64
train_history <- model %>% 
  fit(x = train_x,
      y = train_y,
      batch_size = batch_size,
      epochs = epochs,
      validation_split = 0.1, # 10% validation data
      # print progress but don't create graphic
      verbose = 1,
      view_metrics = 0
  )

plot(train_history) +
  geom_line()
##### model evaluation
# pred_test <- model %>% predict(test_x) %>% '>'(0.5) # returns TRUE or FALSE
# pred_test <- model %>% predict(test_x) # returns percent probability

pred_test <- model %>% predict(test_x) %>% '>'(0.5) #%>% k_cast("int32")
head(pred_test, 10)

#confusion matrix of data
decode <- function(x) as.factor(ifelse(x == 0, "No_follow_up", "Follow_Up"))
pred_class <- decode(pred_test)
true_class <- decode(data_test$Follow_Up)
# Confusion Matrix
table("Prediction" = pred_class, "Actual" = true_class)


plt <- as.data.frame(table("Prediction" = pred_class, "Actual" = true_class))

ggplot(plt, aes(Prediction,Actual, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="#009194") +
  labs(x = "Reference",y = "Prediction") #+
  scale_x_discrete(labels=c("Class_1","Class_2","Class_3","Class_4")) +
  scale_y_discrete(labels=c("Class_4","Class_3","Class_2","Class_1"))

#evaluating the model
data.frame(
  Accuracy = accuracy_vec(pred_class, true_class),
  Recall = sens_vec(pred_class, true_class),
  Precision = precision_vec(pred_class, true_class),
  F1 = f_meas_vec(pred_class, true_class)
)

# placing data back into original data
data_test$prediction <- pred_class

#THIS IS ONLY A SUBSET: need to alter code for real inputs
writexl::write_xlsx(data_test, "data/df_prediction.xlsx", col_names = T)




















