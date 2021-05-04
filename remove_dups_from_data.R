library("dplyr")

data <- read.csv('/home/alexander/projects/car_data_analysis/final_project_data/data.csv', stringsAsFactors = TRUE)
data <- na.omit(data)
a <- subset(data, select=-c(id, desc))
a <- unique(a)
data <- data[rownames(a),]
data <- data %>% 
  group_by(make) %>%
  filter(n() > 25)
write.csv(data, "/home/alexander/projects/car_data_analysis/final_project_data/deduped_data.csv", row.names = FALSE)
