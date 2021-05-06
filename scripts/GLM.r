args <- commandArgs(trailingOnly = TRUE)
job_id <- args[1]
array_idx <- args[2]
data_dir <- args[3]
save_dir <- args[4]

data_file <- file.path(data_dir, "deduped_data.csv")

# Should be proper feature - comment/uncomment for now
# # Ensembling needs consistent train
file_idx <- as.integer(array_idx) %/% 6
paste("loading from", file_idx)
train_ids <- file.path(data_dir, "../ensemble_indices", paste("train_", file_idx, ".csv", sep=""))
test_ids <- file.path(data_dir, "../ensemble_indices", paste("test_", file_idx, ".csv", sep=""))
train_ids <- read.csv(train_ids)
test_ids <- read.csv(test_ids)

# Load libraries

library(caret)
library(doMC)
library(dplyr)

data.df <- read.csv(data_file)
data.df <- data.df %>% 
  mutate(age=2021-manufactured.year) %>%
  select(-manufactured.year)
data.df$make <- as.factor(data.df$make)

str(data.df)

# testIndex <- createDataPartition(data.df$price, times=1, p=0.3, list=F)
# train.df <- data.df[-testIndex,] %>% select(-id) %>% select(-desc)
# test.df <- data.df[testIndex,]
# test_flag <- as.integer(1:nrow(data.df) %in% testIndex)
train.df <- merge(data.df, train_ids, by="id")
test.df <- merge(data.df, test_ids, by="id")
train.df$test <- 0
test.df$test <- 1
data.df <- rbind(train.df, test.df)

str(data.df)
str(train.df)

registerDoMC(8)
getDoParWorkers()

train_ctrl_1 <- trainControl(method="cv",
                           number = 5)

x <- Sys.time()
fit <- train(price ~ ., 
                    data = train.df,
                    # type="Regression",
                    method = "glm", 
                    trControl = train_ctrl_1,
                    family = poisson(link="log")
                    )
end <- Sys.time()
elapsed_time_s <- as.double(round(difftime(end, x, units="secs")))
data.df$prediction <- predict(fit, data.df)
filename <- paste(elapsed_time_s,  ".csv", sep="")
filename <- paste(job_id, array_idx, filename, sep="_")
to_save <- data.df[c("id", "price", "prediction", "test")]
save_dir <- file.path(save_dir, "GLM")
dir.create(save_dir, showWarnings = FALSE)
filename <- file.path(save_dir, filename)
filename
str(to_save)
write.csv(to_save, filename, row.names=FALSE)