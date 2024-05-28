import cv2
import numpy as np


# 元の画像に対して，座標に基づいて番号をテキストで書き込む
# 矩形を描く
def draw_translucent_rect(img, x, y, w, h, color):
    sub_img = img[y : y + h, x : x + w]
    back_rect = np.zeros_like(sub_img, dtype=np.uint8)
    back_rect[:] = (*color, 255) if sub_img.shape[2] == 4 else color
    rect = cv2.addWeighted(sub_img, 0.3, back_rect, 0.7, 1.0)
    img[y : y + h, x : x + w] = rect
    return img


# def draw_translucent_rect(img, x, y, w, h):
#     sub_img = img[y:y+h, x:x+w]
#     black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
#     rect = cv2.addWeighted(sub_img, 0.3, black_rect, 0.7, 1.0)
#     img[y:y+h, x:x+w] = rect
#     return img


# 半透明な矩形の上に文字列を描く
# 高さ (height) はピクセル数で指定する
# paddingで余白部分のサイズを指定する
def draw_text_with_box(
    image,
    text,
    org,  # y, x
    padding=5,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    auto_height=False,
    height=20,
    color=(255, 0, 0),
    thickness=2,
    background_color=(0, 0, 0),
):
    if auto_height:
        height = image.shape[0] // 50
    scale = cv2.getFontScaleFromHeight(font, height, thickness)
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    rect_x = org[1] - padding
    rect_y = org[0] - height - padding
    rect_w = size[0] + (padding * 2)
    rect_h = size[1] + baseline + (padding * 2)
    rect_x = max(rect_x, 0)
    rect_y = max(rect_y, 0)
    max_w = image.shape[1] - rect_x
    max_h = image.shape[0] - rect_y
    rect_w = min(rect_w, max_w)
    rect_h = min(rect_h, max_h)
    draw_translucent_rect(image, rect_x, rect_y, rect_w, rect_h, background_color)
    cv2.putText(image, text, org[::-1], font, scale, color, thickness)
    return image


import base64
from mimetypes import guess_type


# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"


def numpy_image_to_data_url(image: np.ndarray):
    # Convert the image to PNG format in memory
    _, buffer = cv2.imencode(".png", image)
    # Encode the image array into base64
    base64_encoded_data = base64.b64encode(buffer).decode("utf-8")
    # Construct the data URL
    return f"data:image/png;base64,{base64_encoded_data}"
