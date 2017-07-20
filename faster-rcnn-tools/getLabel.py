#coding=utf-8
import os
import os.path
import shutil
import time, datetime
rootdir =  '/home/stefan/work/train'
mvtodir = '/home/stefan/caffe/data/face'

i=0
for parent,dirnames,filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含cd
    f = file("person_train.txt","w+")
    for filename in filenames:
        li = [filename," ","1\n"]
	f.writelines(li)
        i = i + 1
	print (i);
    f.close()

#print(i)
#print("idle :%d", idle)
#print("normal :%d", normal)
#print("crowd :%d", i-idle-normal)
print('over')
