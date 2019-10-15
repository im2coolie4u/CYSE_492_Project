import sys
import os
import csv
import re

#for Kaggle:
#use email: billkelly694@gmail.com
#and password: Pass//word1234
#user: billkelly694
#name: Bill Kelly

def check_exist(filename):
    if not path.exists(filename):
        return False
    else:
        return True

def file_grabber():
    inputs = sys.argv
    user_file = input("please enter a unique file name:")
    if '.csv' not in user_file:
        print("incorrect format specified, appending '.csv',\n Ctrl+C if this is incorrect")
        user_file = user_file + '.csv'
    for i in range(len(inputs)):
        print(inputs[i])
    if len(inputs) < 2:
        print("HELP SCREEN HERE!")
    input_file = sys.argv[1]
    if input_file == '--help':
        print("HELP SCREEN HERE!")
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        line = 0
        user_dupes_count = 0
        for row in csv_file:
            if line == 0:
                print("Options are: " + row)
                contents = re.findall(r'(?!")(\w+)', row)
                print("Options are: ")
                for i in range(len(contents)):
                    print('['+ str(i) + ']: ' + contents[i])
                get_selection = input("Please enter which columns you would like added \nto your file " + user_file + "; \nPlease use commas(,) to separate the inputs:")
                print("Adding the following database contents:")
                selection_enum = get_selection.split(',')
                selection_list = []
                for j in selection_enum:
                    print('[' + str(j) + ']' + contents[int(j)])
                    selection_list.append(contents[int(j)])
                    write_file(selection_list, input_file, user_file)
                    pass
                line += 1
        #print("row = " + str(row_count) + ", " + str(col_count) + ".")
        print('Processed ' + str(row_count + 1) + ' lines.')

def write_file(columns, in_file, out_file):
    header = columns
    print(header)
    exists = check_exist(out_file)
    with open(in_file, 'w', newline='') as csv_file:
        csv_write = csv.writer(csv_file, delimiter=',', quotechar = '"', quoting=csv.QUOTE_MINIMAL)
        if exists == False:
            csv_write.writerow(columns)
        else:
            csv.write.writerow()


def main():
    file_grabber()
    print('main!')
#do stuff

if __name__ == '__main__':
    main()
