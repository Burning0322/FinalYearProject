import requests
from bs4 import BeautifulSoup

url = "https://blog.csdn.net/weixin_42357472/article/details/127994090"
headers = {"User-Agent": "Mozilla/5.0"}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# 获取文章内容
article = soup.find("div", {"id": "content_views"})
if article:
    print(article.get_text())
else:
    print("未找到文章内容")
