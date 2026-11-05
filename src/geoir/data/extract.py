import hashlib
import urllib.request
from pathlib import Path

from pyrosm import OSM

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"

PBF_URL = "https://download.geofabrik.de/europe/bosnia-herzegovina-latest.osm.pbf"
PBF_PATH = RAW_DIR / "bih.osm.pbf"
MD5_PATH = RAW_DIR / "bih.osm.pbf.md5"
OUT_PATH = INTERIM_DIR / "pois.parquet"


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")   # nepotpun download nikad ne završi pod pravim imenom
    print(f"Skidam {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)


def ensure_data() -> None:
    # 1. struktura foldera
    for d in (RAW_DIR, INTERIM_DIR):
        if not d.exists():
            print(f"Kreiram {d}")
            d.mkdir(parents=True, exist_ok=True)

    # 2. PBF fajl
    if not PBF_PATH.exists():
        download(PBF_URL, PBF_PATH)
        download(PBF_URL + ".md5", MD5_PATH)

    # 3. provjera integriteta
    if MD5_PATH.exists():
        expected = MD5_PATH.read_text().split()[0].lower()
        actual = md5_of(PBF_PATH)
        if actual != expected:
            raise RuntimeError(
                f"md5 se ne poklapa (očekivano {expected}, dobijeno {actual}). "
                f"Obriši {PBF_PATH} i pokreni ponovo."
            )
        print("md5 OK")


if __name__ == "__main__":
    ensure_data()

    osm = OSM(str(PBF_PATH))
    gdf = osm.get_pois()
    print(len(gdf))
    print(gdf["osm_type"].value_counts())
    print(gdf.total_bounds)
    gdf.to_parquet(OUT_PATH)