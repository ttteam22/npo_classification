from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
import re

def get_links(url):
  html = urlopen(url).read()
  soup = BeautifulSoup(html)
  links = []
  for link in soup.findAll('a'):
      links.append(link.get('href'))
  return links

def get_text_list():
  url = "http://oppid.ru/"
  html = urlopen(url).read()
  soup = BeautifulSoup(html)
  data = soup.findAll(text=True)
  data_filtered = [each for each in data if each!="\n" and each!=" "]
  return data_filtered