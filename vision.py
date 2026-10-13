"""Image handling: shrink uploads and encode them for the vision model."""

import base64
import io

from PIL import Image


def to_data_url(image_bytes: bytes, mime: str) -> str:
    """Downscale large photos and return a base64 data URL.

    Phone photos can be several megabytes; resizing keeps requests fast and well
    under Groq's size limit. Falls back to the original bytes if processing fails.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.thumbnail((1600, 1600))
        keep_png = mime == "image/png" and image.mode in ("RGBA", "LA", "P")
        buffer = io.BytesIO()
        if keep_png:
            image.save(buffer, format="PNG")
            out_mime = "image/png"
        else:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.save(buffer, format="JPEG", quality=88)
            out_mime = "image/jpeg"
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return f"data:{out_mime};base64,{encoded}"
    except Exception:
        return f"data:{mime};base64,{base64.b64encode(image_bytes).decode()}"