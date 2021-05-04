data <- read.csv('/home/alexander/projects/car_data_analysis/final_project_data/data.csv')
data <- na.omit(data)
incomparables = c(id, desc)
a <- subset(data, select=-c(id, desc))
a <- unique(a)
data <- data[rownames(a),]
