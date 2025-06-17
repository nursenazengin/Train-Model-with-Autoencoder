import random
import csv

a = random.uniform(50, 100)
b = random.uniform(20, 40)
c = random.uniform(500, 1000)
d = random.uniform(0, 10)
e = random.uniform(30, 70)
f = random.uniform(5, 19)

list = []

list.append(a)
list.append(b)
list.append(c)
list.append(d)
list.append(e)
list.append(f)

for i in range(len(list)):
  print(list[i])

with open('../data/predict.csv', mode='a', newline='') as file:
  writer =csv.writer(file)
  writer.writerow(list)