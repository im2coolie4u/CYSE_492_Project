import sys
import os
import csv

#for Kaggle:
#use email: billkelly694@gmail.com
#and password: Pass//word1234
#user: billkelly694
#name: Bill Kelly

def file_grabber():
    inputs = sys.argv
    for i in range(len(inputs)):
        print(inputs[i])
    if len(inputs) < 2:
        print("HELP SCREEN HERE!")
    input_file = sys.argv[1]
    if input_file == '--help':
        print("HELP SCREEN HERE!")
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        row_count = 0
        col_count = 0
        line = 0
        user_list = []
        user_dupes_count = 0
        user_dupes = []
        for row in csv_file:
            row = row.join(',')
            if line == 0:
                print("Options are: ")
                for col in row:
                    print("[" + col +  "]" + ": ")
            line += 1
                #print("row = " + str(row_count) + ", " + str(col_count) + ".")
        print('Processed ' + str(row_count + 1) + ' lines.')
        print(line)
        print(user_dupes_count)


def main():
    file_grabber()
    print('main!')
#do stuff

if __name__ == '__main__':
    main()
