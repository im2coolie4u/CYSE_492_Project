import csv

lines = [['Bob', 'male', '27'],
['Smith', 'male', '26'],
['Alice', 'female', '26']]

header = ['name', 'gender', 'age']

with open("test.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) # write the header
    # write the actual content line by line
    for l in lines:
        writer.writerow(l)
    # or we can write in a whole
    # writer.writerows(lines)
