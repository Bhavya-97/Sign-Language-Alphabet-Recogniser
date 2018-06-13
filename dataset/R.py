import os
path = '.'


alpha = ['B']
for j in alpha:
    path='./'+j+'/';
    files = os.listdir(path)
    i = 501
    for filem in files:
        
        os.rename("./"+j+"/"+filem, "./"+j+"/"+j+str(i)+'.jpg')
        i = i+1
