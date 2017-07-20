#!/usr/bin/python
#coding=utf-8

from xml.dom.minidom import parse
import xml.dom.minidom
from xml.dom.minidom import Document
import codecs
import os
import os.path
import sys

rootdir = r'/home/stefan/Annotations'
resdir = r'/home/stefan/newAnnotations'

print "ok"

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:
        str_text = rootdir+'/'+filename
        domTree = xml.dom.minidom.parse(str_text)
        data = domTree.documentElement
        name = data.getElementsByTagName("filename")
        name_str = ''
        width_str = ''
        height_str = ''
        depth_str = ''
        xmin_str = ''
        ymin_str = ''
        xmax_str = ''
        ymax_str = ''
        for names in name:
            name_str = names.childNodes[0].data

        size = data.getElementsByTagName("size")
        for sizes in size:
            width = sizes.getElementsByTagName("width")[0]
            width_str = width.childNodes[0].data
            height = sizes.getElementsByTagName("height")[0]
            height_str = height.childNodes[0].data
            depth = sizes.getElementsByTagName("depth")[0]
            depth_str = depth.childNodes[0].data

        object_name = []
        leibie = data.getElementsByTagName("object")
        for objects in leibie:
            lei = objects.getElementsByTagName("name")[0]
            lei_str = lei.childNodes[0].data
            #get the obkect name
            object_name.append(lei_str)
            #print lei_str

        print object_name[0]
        print object_name[1]
        print object_name[2]

        box = data.getElementsByTagName("bndbox")
        xmin_str_arr = []
        ymin_str_arr = []
        xmax_str_arr = []
        ymax_str_arr = []
        for boxes in box:
            xmin = boxes.getElementsByTagName("xmin")[0]
            xmin_str = xmin.childNodes[0].data

            #get the xmin_str_arr
            xmin_str_arr.append(xmin_str)

            ymin = boxes.getElementsByTagName("ymin")[0]
            ymin_str = ymin.childNodes[0].data

            #get the ymin_str_arr
            ymin_str_arr.append(ymin_str)

            xmax = boxes.getElementsByTagName("xmax")[0]
            xmax_str = xmax.childNodes[0].data

            #get the xmax_str_arr
            xmax_str_arr.append(xmax_str)

            ymax = boxes.getElementsByTagName("ymax")[0]
            ymax_str = ymax.childNodes[0].data

            #get the ymax_str_arr
            ymax_str_arr.append(ymax_str)


        with codecs.open(resdir + '/' + name_str + '.xml', 'w', 'utf-8')as xml2:
            xml2.write('<annotation verified="no">\n')
            xml2.write('\t<folder>JPEGImages</folder>\n')
            xml2.write('\t<filename>' + name_str + '</filename>\n')
            xml2.write('\t<source>\n')
            xml2.write('\t\t<database>MyFace</database>\n')
            xml2.write('\t</source>\n')
            xml2.write('\t<size>\n')
            xml2.write('\t\t<width>' + width_str + '</width>\n')
            xml2.write('\t\t<height>' + height_str + '</height>\n')
            xml2.write('\t\t<depth>' + depth_str + '</depth>\n')
            xml2.write('\t</size>\n')
            xml2.write('\t<segmented>0</segmented>\n')

            xml2.write('\t<object>\n')
            xml2.write('\t\t<name>'+object_name[0]+'</name>\n')
            xml2.write('\t\t<pose>Unspecified</pose>\n')
            xml2.write('\t\t<truncated>0</truncated>\n')
            xml2.write('\t\t<difficult>0</difficult>\n')
            xml2.write('\t\t<bndbox>\n')
            xml2.write('\t\t\t<xmin>' + xmin_str_arr[0] + '</xmin>\n')
            xml2.write('\t\t\t<ymin>' + ymin_str_arr[0] + '</ymin>\n')
            xml2.write('\t\t\t<xmax>' + xmax_str_arr[0] + '</xmax>\n')
            xml2.write('\t\t\t<ymax>' + ymax_str_arr[0] + '</ymax>\n')
            xml2.write('\t\t</bndbox>\n')
            xml2.write('\t</object>\n')

            xml2.write('\t<object>\n')
            xml2.write('\t\t<name>'+object_name[1]+'</name>\n')
            xml2.write('\t\t<pose>Unspecified</pose>\n')
            xml2.write('\t\t<truncated>0</truncated>\n')
            xml2.write('\t\t<difficult>0</difficult>\n')
            xml2.write('\t\t<bndbox>\n')
            xml2.write('\t\t\t<xmin>' + xmin_str_arr[1] + '</xmin>\n')
            xml2.write('\t\t\t<ymin>' + ymin_str_arr[1] + '</ymin>\n')
            xml2.write('\t\t\t<xmax>' + xmax_str_arr[1] + '</xmax>\n')
            xml2.write('\t\t\t<ymax>' + ymax_str_arr[1] + '</ymax>\n')
            xml2.write('\t\t</bndbox>\n')
            xml2.write('\t</object>\n')

            xml2.write('\t<object>\n')
            xml2.write('\t\t<name>'+object_name[2]+'</name>\n')
            xml2.write('\t\t<pose>Unspecified</pose>\n')
            xml2.write('\t\t<truncated>0</truncated>\n')
            xml2.write('\t\t<difficult>0</difficult>\n')
            xml2.write('\t\t<bndbox>\n')
            xml2.write('\t\t\t<xmin>' + xmin_str_arr[2] + '</xmin>\n')
            xml2.write('\t\t\t<ymin>' + ymin_str_arr[2] + '</ymin>\n')
            xml2.write('\t\t\t<xmax>' + xmax_str_arr[2] + '</xmax>\n')
            xml2.write('\t\t\t<ymax>' + ymax_str_arr[2] + '</ymax>\n')
            xml2.write('\t\t</bndbox>\n')
            xml2.write('\t</object>\n')

            xml2.write('</annotation>')