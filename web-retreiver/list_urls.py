import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

visited_links = set()


def scrape_links(url):
    if url in visited_links:
        return
    print("Scraping:", url)
    visited_links.add(url)

    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to retrieve the page: {url}")
        return

    soup = BeautifulSoup(response.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if href:
            absolute_url = urljoin(url, href)
            if base_url in absolute_url:
                scrape_links(absolute_url)


if __name__ == "__main__":
    base_url = "https://kubernetes.io/docs"
    scrape_links(base_url)
