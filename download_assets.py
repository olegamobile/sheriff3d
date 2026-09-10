import os
import urllib.request

assets = [
    "assets/manual/bequille.jpg",
    "assets/manual/capote.jpg",
    "assets/manual/gouv.jpg",
    "assets/manual/interieur.jpg",
    "assets/manual/pied.jpg",
    "assets/manual/pont.jpg",
    "assets/manual/ris.jpg",
    "assets/manual/spi.gif",
    "assets/manual/vdetail.jpg",
    "assets/manual/voiles.jpg",
    "assets/manual/wc.jpg",
    "assets/photos/Constance_I.jpg",
    "assets/photos/axesafran.jpg",
    "assets/photos/farfelu.jpg",
    "assets/photos/horn.jpg",
    "assets/photos/ladesirade.jpg",
    "assets/photos/mouillage_Aix.jpg",
    "assets/photos/mouro1.jpg",
    "assets/photos/mouro2.jpg",
    "assets/photos/noddi1.jpg",
    "assets/photos/noddi2.jpg",
    "assets/photos/port_RT_Arriere.jpg",
    "assets/photos/port_RT_avant.jpg",
    "assets/photos/port_RT_pont.jpg",
    "assets/photos/samoa_ii.jpg"
]

base_url = "https://sheriff.aland.space/"

req_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

for asset in assets:
    url = base_url + asset
    out_path = os.path.join("downloaded_assets", asset.replace("/", os.sep))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req) as resp, open(out_path, 'wb') as f:
            f.write(resp.read())
        print(f"Downloaded: {asset} ({os.path.getsize(out_path)} bytes)")
    except Exception as e:
        print(f"Failed {asset}: {e}")

