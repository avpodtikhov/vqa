import json

with open("by_question_type.json", "r") as fp:
    by_question_type = json.load(fp)
with open("by_answer_type.json", "r") as fp:
    by_answer_type = json.load(fp)
print(by_answer_type)
print(by_question_type)
plus = 0
tot = 0
labels = []
sizes = []

for key in by_answer_type:
    plus += by_answer_type[key][0] 
    tot += by_answer_type[key][1]
    print(key, by_answer_type[key][0] / (by_answer_type[key][1]))
print('Total', plus / tot)
'''
for key in by_answer_type:
    labels.append(key,)
    sizes.append(by_answer_type[key][1] / (by_answer_type[key][1] + by_answer_type[key][0]))
import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.savefig('stat1.png')
labels = []
sizes = []
import matplotlib
matplotlib.rcParams.update({'font.size': 7})

for key in by_question_type:
    labels.append(key)
    sizes.append(by_question_type[key][1])
import matplotlib.pyplot as plt
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.savefig('stat2.png')
'''