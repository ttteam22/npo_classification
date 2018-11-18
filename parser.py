from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re
from langdetect import detect
import requests
from requests.exceptions import MissingSchema

def get_links(url):
    try:
        links = []
        domain = url.split(".")[0]
        r = requests.get(url)
        html = r.text
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

def link_splitter(url_string):
    result = []
    urls = []
    
    for url in url_string.split(","):
        urls.append(url)
        
    for url in urls:
        for each in url.split("\n"):
            result.append(each)
            
    return result

def get_text_contents(url) -> list:
    try:
        r = requests.get(url, timeout=(5.05, 27))
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
                soup = BeautifulSoup(html)
                data = soup.findAll(text=True)
                if isinstance(data, str):
                    return [data]
                data_filtered = [each.strip() for each in data if each!="\n" and each!=" "]
                data_filtered = [each for each in data_filtered if each!=""]
                return data_filtered
            except BaseExceptions as e:
                print("Error: {} {}".format(e, url))
        
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
