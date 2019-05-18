import requests_html
import pandas as pd
from CrawlingBlog import url_list

# 获取所有的url地址并储存于列表url_list中
def url_all():
    for page in range(1, 2):
        url='http://blog.csdn.net/?ref=toolbar_logo&page='+str(page)
        url_list.append(url)

def get_title_read_from_sel(sel):
    mylist = []
    try:
        results = r.html.find(sel)
        for result in results:
            mytitle = result.text
            # myread = list(result.absolute_links)[0]
            mylist.append(mytitle)#, myread))
        return mylist
    except:
        return None

# url_all()
session = requests_html.HTMLSession()
# for url in url_list:
#     r = session.get(url)
#     sel = '#feedlist_id > li'
#     get_title_read_from_sel(sel)
url = 'http://blog.csdn.net/?ref=toolbar_logo&page=1'
r=session.get(url)
sel = '#feedlist_id > li'
print(get_title_read_from_sel(sel))
# print(r.html.text)
# r.html.links
# r.html.absolute_links
# sel = 'body > div.note > div.post > div.article > div.show-content > div > p:nth-child(4) > a'
# results = r.html.find(sel)
# results
# results[0].text
# results[0].absolute_links
# list(results[0].absolute_links)[0]
# print(get_text_link_from_sel(sel))
# sel = 'body > div.note > div.post > div.article > div.show-content > div > p:nth-child(6) > a'
# print(get_text_link_from_sel(sel))
# sel = 'body > div.note > div.post > div.article > div.show-content > div > p > a'
# print(get_text_link_from_sel(sel))
# df = pd.DataFrame(get_text_link_from_sel(sel))
# df.columns = ['text', 'link']
# df.to_csv('output.csv', encoding='gbk', index=False)

