import pandas as pd

from .config import CSV_PATH, IMAGE_DIR, MASK_DIR, CLASSES


def load_dataset():

    # CSV'yi oku
    df = pd.read_csv(CSV_PATH)

    # Gerekli sütunları kontrol et
    required = ["image"] + CLASSES

    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Eksik sütunlar: {missing_cols}")

    # Görüntü ID kontrolü
    if df["image"].isna().any():
        raise ValueError("Eksik görüntü ID'si var.")

    if df["image"].duplicated().any():
        raise ValueError("Tekrarlanan görüntü ID'si var.")

    # Sınıf doğrulama
    if not df[CLASSES].notna().all().all():
        raise ValueError("Eksik sınıf etiketleri var.")

    if not df[CLASSES].isin([0, 1]).all().all():
        raise ValueError("Sınıf etiketleri 0 veya 1 değil.")

    if not (df[CLASSES].sum(axis=1) == 1).all():
        raise ValueError("Geçersiz one-hot sınıf etiketleri var.")

    # Hedef sınıfı oluştur
    df["label"] = df[CLASSES].idxmax(axis=1)

    # Görüntü yollarını oluştur
    df["image_path"] = df["image"].apply(
        lambda image_id: IMAGE_DIR / f"{image_id}.jpg"
    )

    # Görüntüleri kontrol et
    missing_images = ~df["image_path"].map(
        lambda p: p.is_file()
    )

    if missing_images.any():
        raise ValueError(
            f"Eksik görüntü sayısı: {missing_images.sum()}"
        )

    # Maske sözlüğünü oluştur
    mask_lookup = {
        p.stem.replace("_segmentation", ""): p
        for p in MASK_DIR.iterdir()
        if p.is_file() and "_segmentation" in p.stem
    }

    # Maskeleri eşleştir
    df["mask_path"] = df["image"].map(mask_lookup)

    return df