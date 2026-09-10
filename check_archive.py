import urllib.request
import json

urls_to_test = [
    "http://sheriff600.chez.com/plans/plans.html",
    "http://sheriff600.chez.com/idees/idees.html",
    "http://sheriff600.chez.com/sheriff/manuel/manuel.html",
    "http://sheriff600.chez.com/sheriff/article/articlebx.html"
]

headers = {'User-Agent': 'Mozilla/5.0'}

for u in urls_to_test:
    # 1. Try directly
    try:
        req = urllib.request.Request(u, headers=headers)
        with urllib.request.urlopen(req, timeout=5) as r:
            print(f"Direct success {u}: status {r.status}")
    except Exception as e:
        print(f"Direct failed {u}: {e}")
        # 2. Try Wayback Machine API
        wb_api = f"https://archive.org/wayback/available?url={u}"
        try:
            req2 = urllib.request.Request(wb_api, headers=headers)
            with urllib.request.urlopen(req2, timeout=5) as r2:
                data = json.loads(r2.read().decode('utf-8'))
                snapshots = data.get('archived_snapshots', {})
                closest = snapshots.get('closest')
                if closest and closest.get('available'):
                    print(f"  Wayback found for {u}: {closest.get('url')}")
                else:
                    print(f"  Wayback NOT found for {u}")
        except Exception as e2:
            print(f"  Wayback error: {e2}")
