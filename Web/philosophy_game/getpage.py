#!/usr/bin/python3
# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
from json import loads
from urllib.request import urlopen
from urllib.parse import urlencode, unquote
import ssl
from collections import OrderedDict
from flask import Flask

app = Flask(__name__)


global cache
cache = dict()


def formatCorrection(txt) :
    return unquote(txt).split('#')[0].replace('_', ' ')

def getJSON(page):
    params = urlencode({
      'format': 'json',
      'action': 'parse',
      'prop': 'text',
      'redirects': 'true',
      'page': page})
    API = "https://fr.wikipedia.org/w/api.php"
    gcontext = ssl.SSLContext()
    response = urlopen(API + "?" + params, context=gcontext)
    return response.read().decode('utf-8')


def getRawPage(page):
    parsed = loads(getJSON(page))
    try:
        title = parsed['parse']['title']
        content = parsed['parse']['text']['*']
        return title, content
    except KeyError:
        # La page demandée n'existe pas
        return None, None

def getHRef(hpage) :
    soup = BeautifulSoup(hpage, 'html.parser')
    soup = soup.find('div')
    href_list = []
    for link in soup.findAll('p', recursive = False) :
        for l in link.findAll('a') :
            href = l.get('href')
            if href != None and href[:6] == '/wiki/' :
                href_list.append(href[6:])
    return href_list[:10]

def getPage(page):
    if page in cache.keys() :
        return page, cache[page]

    try :
        title = formatCorrection(getRawPage(page)[0])
        hrefs = getHRef(getRawPage(page)[1])

        hrefs = [formatCorrection(txt) for txt in hrefs]
        hrefs = list(OrderedDict.fromkeys(hrefs))

        if isinstance(hrefs, str) :
            hrefs = [hrefs]
            # app.logger.info("hrefs " + hrefs)

        cache[page] = hrefs
        return title, hrefs
    except :
        return (None, [])

if __name__ == '__main__':
    # Ce code est exécuté lorsque l'on exécute le fichier

    # Voici des idées pour tester vos fonctions :
    # print(getJSON("Utilisateur:A3nm/INF344"))
    # print(getRawPage("Utilisateur:A3nm/INF344"))
    print(getPage("Sciences_sociales"))
