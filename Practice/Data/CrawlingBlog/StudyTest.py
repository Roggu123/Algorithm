# ------------------------------------------------------------
#                           3.26前的主要参考代码
# ------------------------------------------------------------
# import requests
# import bs4
# from bs4 import BeautifulSoup
# from time import sleep
# import os
# import pandas as pd
#
# # 获取所有页面url地址并储存于列表url_list中
# def url_all():
#     for page in range(1, 2):
#         url='http://blog.csdn.net/?ref=toolbar_logo&page='+str(page)
#         url_list.append(url)
#
# # 获取页面包含的所有博客地址
# def essay_url():
#     blog_urls = []
#     for url in url_list:
#         html = requests.get(url, headers=headers)
#         # text 则是根据设置的encoding来解码,编码是通过chardet.detect来获取的(既使用apparent_encoding)
#         # html.encoding = html.apparent_encoding
#         # html.encoding = ('utf-8','ignore')
#         html.encoding = ('utf_8','ignore')
#         # 使用python标准库解释器'html.parser'来解析html.text
#         soup = BeautifulSoup(html.text, 'html.parser')
#         for h2 in soup.find_all('h2'):
#             blog_url = (h2('a')[0]['href'])
#             blog_urls.append(blog_url)
#     return blog_urls
#
# # 设置爬取数据保存路径
# def save_path():
#     s_path='./'
#     if not os.path.isdir(s_path):
#         os.mkdir(s_path)
#     else:
#         pass
#     return s_path
#
#
# # 找到并保存所有文章标题，内容
# def save_essay(blog_urls, s_path):
#     for blog_url in blog_urls:
#         blog_html = requests.get(blog_url, headers=headers)
#         blog_html.encoding = blog_html.apparent_encoding
#         soup = BeautifulSoup(blog_html.text, 'html.parser')
#         try:
#             # for title in soup.find('h2',{'class':'title-article'}):
#             for title in soup.find('h2'):
#                 if isinstance(title, str):
#                     print('-----文章标题-----：', title)
#                     print('-----文章链接-----：', title.link)
#                     blogname = title
#                     blogname = blogname.replace("\n", '')
#                     blogname = blogname.replace("\r", '')
#                     blogname = blogname.replace(" ", '')
#                     list(blogname)
#                     print("收集标题")
#                     blognames.append(blogname)
#
#             for read in soup.find('',{'class':'read-count'}):
#                 if isinstance(read, str):
#                     print(read)
#                     readnum = read
#                     readnum = readnum.replace("阅读数：", '')
#                     list(readnum)       # readnum是string,要修改格式为列表便于储存
#                     print("收集阅读数")
#                     readnums.append(readnum)
#
#         except BaseException as b:
#             print(b)
#     print('-------------------所有页面遍历完成')
#     return soup
#
# url_list = []
# blogs = []
# blognames = []
# readnums = []
# print("执行StudyTest")
# headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
# print("获取所有页面的url地址并储存于列表url_list中:")
# url_all()
# print("获取页面包含的所有博客的地址")
# essay_url()
# print("输出所有爬取页面的URL")
# print(url_list)
# print("输出爬取页面包含的博客URL")
# print(essay_url())
# print("设置爬取数据的保存路径")
# save_path()
# print("保存爬取标题及阅读数至列表")
# save_essay(essay_url(), save_path())
# print("标题和阅读数存入EXCEl表格")
# blogs.append((blognames,readnums))
# df = pd.DataFrame(blogs)
# df.columns = ['title','num']
# print(df)
# df.to_csv('record.csv', encoding=('gbk'), index=False)
# # df.to_csv('record.csv', index=False)


# # ------------------------------------------------------------
# #                           3.26后主要参考代码
# # ------------------------------------------------------------
# import ssl
#
# from bs4 import BeautifulSoup
# from urllib import request
# import chardet
#
# ssl._create_default_https_context = ssl._create_unverified_context
# url = "https://www.csdn.net/?ref=toolbar_logo&page=1"
# response = request.urlopen(url)
# html = response.read()
# charset = chardet.detect(html)
# # 报错UnicodeDecodeError: 'charmap' codec can't decode byte 0x8e in position 80: character maps to <undefined>
# # 该异常表示python3代码对于一些字符的识别报错
# # 尝试更改html网页编码方式为utf-8,无报错
# html = html.decode(str(charset["encoding"]))  # 设置抓取到的html的编码方式
# # html =html.decode('utf-8')
#
# # 使用剖析器为html.parser
# soup = BeautifulSoup(html, 'html.parser')
# # 获取到每一个class=list_con的a节点
# allList = soup.select('.list_con')
# #遍历列表，获取有效信息
# print("网页链接为:", url)
# for news in allList:
#     number = news.select('.num')
#     author = news.select('.name')
#     eassy_title = news.select('a')
#     # 只选择长度大于0的结果
#     print("eassy tiele:",eassy_title)
#     if len(eassy_title) > 0:
#         # 文章链接
#         try:#如果抛出异常就代表为空
#             href = eassy_title[0]['href']
#         #   href = url + aaa[0]['href']
#         except Exception:
#             href=''
#         # 博客标题
#         try:
#             # title = aaa[0]['title']
#             title = eassy_title[0].text
#         except Exception:
#             title = "标题为空"
#         # 博客阅读数
#         try:
#             readnum = number[0].text
#         except Exception:
#             readnum = "未知阅读数"
#         # 博客作者
#         try:
#             who = author[0].text
#         except Exception:
#             who = "无名氏"
#         print("标题", title, "\nurl：", href, "\n阅读量：", readnum, "\n作者：", who)
#         print("==============================================================================================")


# ------------------------------------------------------------
#                      探究如何把数据存入表格
# ------------------------------------------------------------
# @Time     创建时间 : 2019-03-29 11:00
# @Log      代码说明：如何把网站爬取的数据存到csv表格中
#
# # **************************方法一***************************
# # @Log      代码说明：这段代码可以把数据存储到csv表格，但是出现乱码，如果删掉print里的中文则无乱码
# year = 1
# years = 5
# bj = 10000
# rate = 0.05
#
# f = open("interest.csv", 'w+')
# while year < years:
#     bj = bj * (1 + rate)
#     print("时间{0},本息总和{1}".format(year, bj), file=f)
#     year += 1

# # **************************方法二***************************
# # @Reference   参考 : https://zhuanlan.zhihu.com/p/33085521
# # @Log      代码说明 : 这段代码就是垃圾，没有定义set_style，没法运行
# import xlwt # 载入xlwt数据库
#
# f = xlwt.Workbook() # 创建工作簿
# sheet1 = f.add_sheet(u'sheet1',cell_overwrite_ok=True) # 创建sheet
# row0 = [u'业务',u'状态',u'北京',u'上海',u'广州',u'深圳',u'状态小计',u'合计']
# column0 = [u'机票',u'船票',u'火车票',u'汽车票',u'其它']
# status = [u'预订',u'出票',u'退票',u'业务小计']
#
# # 生成第一行
# for i in range(0,len(row0)):
#   sheet1.write(0,i,row0[i],set_style('Times New Roman',220,True))
#
# # 生成第一列和最后一列(合并4行)
# i, j = 1, 0
# while i < 4*len(column0) and j < len(column0):
#  sheet1.write_merge(i,i+3,0,0,column0[j],set_style('Arial',220,True)) # 第一列
#  sheet1.write_merge(i,i+3,7,7) # 最后一列"合计"
#  i += 4
#  j += 1
#
# sheet1.write_merge(21,21,0,1,u'合计',set_style('Times New Roman',220,True))
#
# # 生成第二列
# i = 0
# while i < 4*len(column0):
#  for j in range(0,len(status)):
#    sheet1.write(j+i+1,1,status[j])
#  i += 4
#
# f.save('demo1.xlsx') # 保存文件

# # **************************方法三***************************
# # @Reference   参考 : https://zhuanlan.zhihu.com/p/33085521
# # @Log      代码说明 : 这段代码可以运行通过，结果也正确
# from bs4 import BeautifulSoup  # 导入BeautifulSoup模块。注意大小写
# import pandas as pd
#
# soup = BeautifulSoup(open('DATA.xml', 'rb'), 'xml', from_encoding='utf-8')  # 按照utf-8编码制度读取xml类型的文件
#
# X = []
# Z = []
# for i in soup.find_all('X'):  # 循环遍历所有标签为X的数据
#     X.append(i.string)  # 将标签数据的string/comment写入到X这个列表中
# for j in soup.find_all('Z'):
#     Z.append(j.string)
#
# # 将列表a，b转换成字典后，把结果转换成DataFrame并保存到Excel中
# c={"X" : X,
#    "Z" : Z}  # 转换为字典
# df = pd.DataFrame(c)
# df.to_excel('DATA.xlsx')

# # ------------------------------------------------------------
# #                 3.26后参考代码修改版，符合我的要求
# # ------------------------------------------------------------
# import ssl
# from bs4 import BeautifulSoup
# from urllib import request
# import chardet
# import xlwt
# import pandas as pd
#
# ssl._create_default_https_context = ssl._create_unverified_context
#
# X = [] # X储存所有链接
# Y = [] # Y储存所有标题
# Z = [] # Z储存所有阅读数
# A = [] # A储存所有作者
#
# #
# for page in range(1, 3):
#     url = "https://www.csdn.net/?ref=toolbar_logo&page=" + str(page)
#     print("遍历第",page,"页\n")
#     response = request.urlopen(url)
#     html = response.read()
#     charset = chardet.detect(html)
#     # 报错UnicodeDecodeError: 'charmap' codec can't decode byte 0x8e in position 80: character maps to <undefined>
#     # 该异常表示python3代码对于一些字符的识别报错
#     # 尝试更改html网页编码方式为utf-8,无报错
#     html = html.decode(str(charset["encoding"]))  # 设置抓取到的html的编码方式
#     # html =html.decode('utf-8')
#
#     # BeatifulSoup是从html或xml文档中提取数据的库，这里选择解析器html.parser
#     # 解析的数据保存至soup
#     soup = BeautifulSoup(html, 'html.parser')
#     # 获取到每一个class=list_con的a节点
#     allList = soup.select('.list_con')
#     #遍历列表，获取有效信息
#     # print("网页链接为:", url)
#     print("网页链接为:", url)
#     for news in allList:
#         number = news.select('.num')
#         author = news.select('.name')
#         eassy_title = news.select('a')
#         # 只选择长度大于0的结果
#         # print("从a节点获取的信息:\n",eassy_title)
#         if len(eassy_title) > 0:
#             # 文章链接
#             try:#如果抛出异常就代表为空
#                 href = eassy_title[0]['href']
#             #   href = url + aaa[0]['href']
#                 X.append(href)
#             except Exception:
#                 href='链接为空'
#             # 博客标题
#             try:
#                 # title = aaa[0]['title']
#                 title = eassy_title[0].text
#                 Y.append(title)
#             except Exception:
#                 title = "标题为空"
#             # 博客阅读数
#             try:
#                 readnum = number[0].text
#                 Z.append(readnum)
#             except Exception:
#                 readnum = "未知阅读数"
#             # 博客作者
#             try:
#                 who = author[0].text
#                 A.append(who)
#             except Exception:
#                 who = "无名氏"
#             print("标题", title, "\nurl：", href, "\n阅读量：", readnum, "\n作者：", who)
#
# print("储存数据至表格.......")
# Table = {"Title":Y,"ReadNumber":Z,"Author":A,"URL":X}
# df = pd.DataFrame(Table)
# df.to_csv('Table.csv',encoding='gbk')
# print("==============================================================================================")

# # ------------------------------------------------------------
# #             3.26后参考代码修改版优化，将各功能模块化
# # ------------------------------------------------------------
import ssl
from bs4 import BeautifulSoup
from urllib import request
import chardet
import xlwt
import pandas as pd

ssl._create_default_https_context = ssl._create_unverified_context
# 遍历csdn的所有页面
def url_all():
   for page in range(1,3):
       url = "https://www.csdn.net/?toolbar_logo&page="+str(page)
       print("将第"+str(page)+"页加入列表url_list")
       url_list.append(url)

def Trans_code(blog_url):
        response = request.urlopen(blog_url)
        html = response.read()
        # 进行编码转换与解码,利用模块chardet可以方便检测网页编码
        charset = chardet.detect(html)
        # 将网页编码转换为str,再进行解码
        html = html.decode(str(charset["encoding"]))
        return html

def Get_data(html):
    soup = BeautifulSoup(html, 'html.parser')
    # 获取到每一个class=list_con的a节点
    allList = soup.select('.list_con')
    # 遍历列表，获取有效信息
    # print("网页链接为:", url)
    print("网页链接为:", blog_url)
    for news in allList:
        number = news.select('.num')
        author = news.select('.name')
        eassy_title = news.select('a')
        # 只选择长度大于0的结果
        # print("从a节点获取的信息:\n",eassy_title)
        if len(eassy_title) > 0:
            # 文章链接
            try:  # 如果抛出异常就代表为空
                href = eassy_title[0]['href']
                #   href = url + aaa[0]['href']
                X.append(href)
            except Exception:
                href = '链接为空'
            # 博客标题
            try:
                # title = aaa[0]['title']
                title = eassy_title[0].text
                Y.append(title)
            except Exception:
                title = "标题为空"
            # 博客阅读数
            try:
                readnum = number[0].text
                Z.append(readnum)
            except Exception:
                readnum = "未知阅读数"
            # 博客作者
            try:
                who = author[0].text
                A.append(who)
            except Exception:
                who = "无名氏"
            print("标题", title, "\nurl：", href, "\n阅读量：", readnum, "\n作者：", who)

def Save_data(X,Y,Z,A):
    print("储存数据至表格.......")
    Table = {"Title": Y, "ReadNumber": Z, "Author": A, "URL": X}
    df = pd.DataFrame(Table)
    df.to_csv('Table.csv', encoding='gbk')
    print("==============================================================================================")

# ---------------------------主程序--------------------------
X = [] # X储存所有链接
Y = [] # Y储存所有标题
Z = [] # Z储存所有阅读数
A = [] # A储存所有作者
url_list = []
url_all()
for blog_url in url_list:
	html = Trans_code(blog_url)
	Get_data(html)
Save_data(X,Y,Z,A)