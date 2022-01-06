import os
import sys
import csv

def intToString(n):
    if n < 10:
        return '00' + str(n)
    elif n < 100:
        return '0' + str(n)
    return n

csvFile = open('./data/train.csv')
reader = csv.reader(csvFile)
next(reader)
landmark = {}
current = 1
i = 1
for row in reader:
    landmarkIndex = int(row[1])
    if landmarkIndex != current:
        current = landmarkIndex
        i = i + 1
    landmark[row[0]] = i


os.chdir('./data')
os.system('rm -r ./images')
os.system('mkdir images')
limit = 10
for i in range(1):
    os.system('bash download.sh train ' + str(i))
    os.system('mkdir images_temp')
    os.system('tar xvf images_' + intToString(i) + '.tar -C ./images_temp')
    os.system('rm images_' + intToString(i) + '.tar')
    os.chdir('./images_temp')
    for folder, _, _ in os.walk('./'):
        if len(folder.split('/')) == 4:
            for image in os.listdir(folder):
                fullName = os.path.basename(image)
                name = os.path.splitext(fullName)[0]
                if name in landmark and landmark[name] <= limit:
                    os.system('mv \'./\'' + image + ' ' + '../images')
    os.chdir('..')
    os.system('rm -r ./images_temp')


"""
current = 1
l = [[]]
for row in reader:
    landmarkIndex = int(row[1])
    if landmarkIndex != current:
        current = landmarkIndex
        l.append([])
    l[len(l) - 1].append(row[0])
"""
