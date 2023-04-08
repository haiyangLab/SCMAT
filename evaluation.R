library(aricode)

answer = read.table(file='E:/Research projects/Subject1/datasets/simulate/3/Label.csv', sep = ',')
result <- read.csv("E:/Research projects/Subject1/datasets/simulate/3/simulate3.SCMAT", sep = ',')


result_vector = as.vector(unlist(result[2]))
answer_vector = as.vector(unlist(answer[2]))


ARI(answer_vector, result_vector)
NMI(answer_vector, result_vector)
