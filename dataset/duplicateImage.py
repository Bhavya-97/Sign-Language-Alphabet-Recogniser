import shutil
import os
path = '.'

alpha = ['E','H','I','L','O','space','W','Y']
for j in alpha:
    path='./'+j+'/';
    files = os.listdir(path)
    i =1001
    for filem in files:
	shutil.copyfile(path+filem, path+j+str(i)+'.jpg')
	i =i + 1
