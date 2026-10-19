
import cv2
import numpy as np


def read_image(path, grayscale=False):
    """
    Görüntüyü Windows dosya yolundaki Türkçe
    karakterlerden etkilenmeden okur.
    """
    flag = (
        cv2.IMREAD_GRAYSCALE
        if grayscale
        else cv2.IMREAD_COLOR
    )

    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8),
        flag
    )

    if image is None:
        raise ValueError(f"Görüntü okunamadı: {path}")

    return image


def resize_image(image, size=(128, 128)):
    """
    Görüntüyü belirtilen boyuta getirir.
    OpenCV'de size = (width, height).
    """
    return cv2.resize(
        image,
        size,
        interpolation=cv2.INTER_AREA
    )


def convert_grayscale(image):
    """
    BGR renkli görüntüyü gri tona dönüştürür.
    """
    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def normalize_image(image):
    """
    uint8 piksel değerlerini 0-1 aralığına getirir.
    """
    return image.astype(np.float32) / 255.0


def preprocess_image(path, size=(128, 128),
                     grayscale=True, normalize=False):

    image = read_image(path)

    image = resize_image(image, size)

    if grayscale:
        image = convert_grayscale(image)

    if normalize:
        image = normalize_image(image)

    return image

def preprocess_with_mask(
    image_path,
    mask_path,
    size=(128, 128),
    grayscale=True,
    normalize=False
):
    # 1. Görüntüyü ve maskeyi oku
    image = read_image(image_path)
    mask = read_image(mask_path, grayscale=True)

    # 2. Görüntüyü yeniden boyutlandır
    image = resize_image(image, size)

    # 3. Maskeyi yeniden boyutlandır
    mask = cv2.resize(
        mask,
        size,
        interpolation=cv2.INTER_NEAREST
    )

    # 4. Maskeyi ikili hale getir
    _, mask = cv2.threshold(
        mask, 127, 255, cv2.THRESH_BINARY
    )

    # 5. Maskeyi görüntüye uygula
    image = cv2.bitwise_and(
        image,
        image,
        mask=mask
    )

    # 6. Gri tonlamaya dönüştür
    if grayscale:
        image = convert_grayscale(image)

    # 7. Normalizasyon
    if normalize:
        image = normalize_image(image)

    return image, mask