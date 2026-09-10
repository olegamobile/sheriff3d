import re
from html.parser import HTMLParser

with open('site.html', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

imgs = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
print("=== IMAGES FOUND ===")
for img in set(imgs):
    print(img)

links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\']', content, re.IGNORECASE)
print("\n=== LINKS FOUND ===")
for l in set(links):
    print(l)
