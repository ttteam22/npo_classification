from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
from langdetect import detect
import requests
from requests.exceptions import MissingSchema, ReadTimeout, ConnectTimeout, ConnectionError

def get_links(url):
    try:
        links = []
        domain = url.split(".")[0]
        r = requests.get(url)
        html = r.text
        soup = BeautifulSoup(html, features="lxml")
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

def link_splitter(url_string):
    result = []
    urls = []
    url_string = url_string.strip()
    for url in url_string.split(","):
        urls.append(url)
        
    for url in urls:
        for each in url.split("\n"):
            result.append(each)
            
    return result

def get_text_contents(url) -> list:
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    try:
        r = requests.get(url, ю)
        html = r.text
        soup = BeautifulSoup(html)
        data = soup.findAll(text=True)
        if isinstance(data, str):
            return [data]
        data_filtered = [each.strip() for each in data if each!="\n" and each!=" "]
        data_filtered = [each for each in data_filtered if each!=""]
        return data_filtered
    
    except MissingSchema:
        prefix = ["http://", "https://"]
        for each in prefix:
            try:
                r = requests.get(each+url, timeout=(5.05, 27))
                html = r.text
                soup = BeautifulSoup(html, features="html5lib")
                data = soup.findAll(text=True)
                if isinstance(data, str):
                    return [data]
                data_filtered = [each.strip() for each in data if each!="\n" and each!=" "]
                data_filtered = [each for each in data_filtered if each!=""]
                return data_filtered
            except BaseException as e:
                print("Error: {} {}".format(e, url))
    except (ReadTimeout, ConnectTimeout, ConnectionError):
        pass
        
def get_filtered_text(url):
    result = []
    print(url)
    try:
        data_filtered = get_text_contents(url)
        for each in data_filtered:
            try:
                if detect(each)=="ru":
                    result.append(each)
            except:
                pass
        return " ".join(result)
    except:
        print("Error: ", url)
        return None
