from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import cyrtranslit
import re
from langdetect import detect
import requests

url = "http://www.rbfondveteranov.ru/"

def get_links(url):
    try:
        links = []
        domain = url.split(".")[0]
        r = requests.get(url)
        html = r.text
        # html = urlopen(url).read()
        soup = BeautifulSoup(html)
        for link in soup.findAll('a'):
            ref = link.get('href')
            try:
                if ref!="/" and ref.startswith(domain):
                    links.append(ref)
            except:
                pass 
        return list(set(links))
    
    except BaseException:
        return None

def get_text_contents(url) -> list:
    # html = urlopen(url).read()
    r = requests.get(url)
    html = r.text
    soup = BeautifulSoup(html)
    data = soup.findAll(text=True)
    if isinstance(data, str):
        return [data]
    data_filtered = [each.strip() for each in data if each!="\n" and each!=" "]
    data_filtered = [each for each in data_filtered if each!=""]
    return data_filtered

def get_filtered_text(url):
    result = []
    data_filtered = get_text_contents(url)
    for each in data_filtered:
        try:
            if detect(each)=="ru":
                result.append(each)
        except:
            pass
    return " ".join(result)


def get_title(url) -> str:
    return data_filtered
