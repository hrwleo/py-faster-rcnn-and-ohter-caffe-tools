#coding=utf-8
import urllib
import  re

def getHtml(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html

def getImg(html):
    reg = r'src="(.+?\.jpg@360h)" lowsrc="'
    imgre = re.compile(reg)
    imglist = re.findall(imgre, html)
    x = 0
    for imgurl in imglist:
	print imgurl
        urllib.urlretrieve(imgurl, 'kouzhao%s.jpg' % x)
        x += 1
    return imglist

html = getHtml("http://www.quanjing.com/category/12705917.html")
#print html
print getImg(html)
