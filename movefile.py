#coding=utf-8
import os
import os.path
import shutil
import time, datetime
rootdir =  '/home/stefan/caffe/data/originalPics'
mvtodir = '/home/stefan/caffe/data/face'
i = 0
idle = 0
normal = 0
crowd = 0
for parent,dirnames,filenames in os.walk(rootdir):    #三个参数：分别返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    #for filename in  filenames:                       #输出文件夹信息
        #print "parent is:" + parent[parent.rfind('\\')+1:]
        #print  "dirname is" + dirname
       
    for filename in filenames:
        #typename = parent[parent.rfind('\\')+1:]
        #if(typename == 'crowd'):
        #crowd = crowd + 1
	#print (filename)
        shutil.copy(os.path.join(parent,filename),  mvtodir+'/'+str(i)+'.jpg')
        #if(typename == "idle"):
            #idle = idle + 1
            #shutil.copy(os.path.join(parent,filename),  mvtodir+'\\idle\\'+ filename)
        #if(typename == "normal"):
            #normal = normal + 1
            #shutil.copy(os.path.join(parent,filename),  mvtodir+'\\normal\\'+ filename)
    
        #print "filename is:" + filename
        #print "the full name of the file is:" + os.path.join(parent,filename) #输出文件路径信息
        i = i + 1
	print (i);

#print(i)
#print("idle :%d", idle)
#print("normal :%d", normal)
#print("crowd :%d", i-idle-normal)
print('over')
