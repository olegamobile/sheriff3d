import urllib.request
import json

base_url = "https://sheriff.aland.space"

# Let's inspect all matches of 'plan', 'wmf', 'dwg', 'dxf', 'pdf', 'jpg', 'png', 'gif' in site.html
with open('site.html', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
matches = re.findall(r'[\'"]([^\'"]*(?:\.jpg|\.png|\.gif|\.pdf|\.wmf|\.dxf|\.dwg|\.svg|plan|sheriff|calife)[^\'"]*)[\'"]', text, re.I)
print("Matches in site.html:")
for m in sorted(set(matches)):
    print(m)

