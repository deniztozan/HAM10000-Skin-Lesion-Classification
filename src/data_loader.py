"""HAM10000 veri setini yükleme ve ortak bütünlük kontrolleri."""

import pandas as pd

from .config import CSV_PATH, IMAGE_DIR, MASK_DIR, CLASSES


def load_dataset(require_masks: bool = False) -> pd.DataFrame:
    """Veri setini doğrulayarak yükle.

    Args:
        require_masks: True ise bütün görüntüler için maske bulunmasını zorunlu tutar.
            Maskesiz modeller için False bırakılabilir.

    Returns:
        Görüntü ID, etiket, image_path ve mask_path içeren DataFrame.
    """
    df = pd.read_csv(CSV_PATH)

    required = ["image", *CLASSES]
    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Eksik sütunlar: {sorted(missing_cols)}")

    if df["image"].isna().any():
        raise ValueError("Eksik görüntü ID'si var.")
    if df["image"].duplicated().any():
        raise ValueError("Tekrarlanan görüntü ID'si var.")

    if not df[CLASSES].notna().all().all():
        raise ValueError("Eksik sınıf etiketleri var.")
    if not df[CLASSES].isin([0, 1]).all().all():
        raise ValueError("Sınıf etiketleri 0 veya 1 değil.")
    if not (df[CLASSES].sum(axis=1) == 1).all():
        raise ValueError("Geçersiz one-hot sınıf etiketleri var.")

    df["label"] = df[CLASSES].idxmax(axis=1)
    df["image_path"] = df["image"].apply(
        lambda image_id: IMAGE_DIR / f"{image_id}.jpg"
    )
    missing_images = ~df["image_path"].map(lambda path: path.is_file())
    if missing_images.any():
        sample = df.loc[missing_images, "image"].head(5).tolist()
        raise ValueError(
            f"Eksik görüntü sayısı: {int(missing_images.sum())}. "
            f"Örnek ID'ler: {sample}"
        )

    # Maskeleri, sadece beklenen *_segmentation.png biçimiyle eşleştir.
    mask_lookup = {}
    if MASK_DIR.is_dir():
        for path in MASK_DIR.glob("*_segmentation.png"):
            image_id = path.name.removesuffix("_segmentation.png")
            if image_id in mask_lookup:
                raise ValueError(f"Tekrarlanan maske ID'si: {image_id}")
            mask_lookup[image_id] = path
    elif require_masks:
        raise FileNotFoundError(f"Maske klasörü bulunamadı: {MASK_DIR}")

    df["mask_path"] = df["image"].map(mask_lookup)

    if require_masks:
        missing_masks = df["mask_path"].isna()
        if missing_masks.any():
            sample = df.loc[missing_masks, "image"].head(5).tolist()
            raise ValueError(
                f"Eksik maske sayısı: {int(missing_masks.sum())}. "
                f"Örnek ID'ler: {sample}"
            )
        unreadable_masks = ~df["mask_path"].map(lambda path: path.is_file())
        if unreadable_masks.any():
            raise ValueError(
                f"Erişilemeyen maske sayısı: {int(unreadable_masks.sum())}"
            )

    return df
