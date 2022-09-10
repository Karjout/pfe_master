# karjout abdeslam
import requests
from bs4 import BeautifulSoup


def get_html(url):
    response = requests.get(url)
    if not response.ok:
        print(f'Code :{response.status_code},url:{url}')
    return response.text


def get_data(html):
    soup = BeautifulSoup(html, 'lxml')
    Links = []
    for a in soup.find_all('a', attrs={'class': 'product-finder-result__link'}, href=True):
        Links.append(a['href'])
    with open('links.txt', 'w') as f:
        for i in Links:
            f.write("%s\n" % i)
    f.close()
    return Links


def main():
    url = 'http://www.google.com?'
    data = get_data(get_html(url))


if __name__ == '__main__':
    main()
