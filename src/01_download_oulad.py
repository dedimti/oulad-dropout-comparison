"""
01_download_oulad.py
Mengunduh dan memverifikasi dataset OULAD asli (Open University, CC-BY 4.0).
Sumber resmi: https://analyse.kmi.open.ac.uk/open_dataset
"""
import os
import sys
import zipfile
import io

DATA_DIR = "data"
# URL resmi Open University. Jika berubah/diblokir, lihat "Unduh manual" di README.
OULAD_URL = "https://analyse.kmi.open.ac.uk/open_dataset/download"

REQUIRED_FILES = [
    "studentInfo.csv", "studentVle.csv", "studentAssessment.csv",
    "assessments.csv", "studentRegistration.csv", "courses.csv", "vle.csv",
]


def already_present():
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_FILES)


def try_download():
    import requests
    from tqdm import tqdm
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Mengunduh OULAD dari {OULAD_URL} ...")
    try:
        r = requests.get(OULAD_URL, stream=True, timeout=120)
        r.raise_for_status()
    except Exception as e:
        print(f"\n[GAGAL] Unduhan otomatis tidak berhasil: {e}")
        print("Silakan unduh manual sesuai bagian 'Unduh manual' di README.md")
        return False

    total = int(r.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            buf.write(chunk)
            bar.update(len(chunk))
    buf.seek(0)
    try:
        with zipfile.ZipFile(buf) as z:
            z.extractall(DATA_DIR)
    except zipfile.BadZipFile:
        print("\n[GAGAL] File terunduh bukan ZIP valid. Unduh manual via README.md")
        return False
    return True


def verify():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"[KURANG] File belum lengkap: {missing}")
        return False
    import pandas as pd
    si = pd.read_csv(os.path.join(DATA_DIR, "studentInfo.csv"))
    print(f"\n[OK] studentInfo.csv: {len(si):,} baris, kolom: {list(si.columns)}")
    print(f"[OK] Distribusi final_result:\n{si['final_result'].value_counts()}")
    print("\nDataset OULAD asli siap. Lanjut: python 02_run_experiment.py")
    return True


if __name__ == "__main__":
    if already_present():
        print("[INFO] File OULAD sudah ada di folder data/. Melewati unduhan.")
        verify()
        sys.exit(0)
    if try_download():
        verify()
    else:
        sys.exit(1)
