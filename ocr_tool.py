#!/usr/bin/env python3
"""
High-Accuracy OCR Tool
======================
Extracts text from any image or PDF file using multiple OCR engines
and advanced image preprocessing for maximum accuracy.

Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP, GIF, PDF

Engines used:
  1. Tesseract OCR (with multiple PSM modes)
  2. EasyOCR (deep learning based)

Preprocessing pipeline:
  - Upscaling (for low-res images)
  - Grayscale conversion
  - Denoising (Non-local Means)
  - Adaptive thresholding
  - Deskewing
  - Sharpening
  - Morphological cleanup

Usage:
  python ocr_tool.py <file_path> [options]

Options:
  --engine tesseract|easyocr|both   OCR engine to use (default: both)
  --lang LANG                       Language code (default: en)
  --preprocess                      Force preprocessing (default: auto)
  --no-preprocess                   Skip preprocessing
  --pages PAGES                     PDF pages to process, e.g. "1-3,5" (default: all)
  --output FILE                     Write output to file instead of stdout
  --confidence                      Show per-line confidence scores
  --verbose                         Show detailed processing info
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

warnings.filterwarnings("ignore")

# Check which engines are available
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    details: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Image preprocessing pipeline
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """Advanced image preprocessing for optimal OCR accuracy."""

    @staticmethod
    def upscale(img: np.ndarray, min_height: int = 1500) -> np.ndarray:
        """Upscale small images so OCR engines can read them clearly."""
        h, w = img.shape[:2]
        if h < min_height:
            scale = min_height / h
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return img

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def denoise(img: np.ndarray) -> np.ndarray:
        """Non-local means denoising — removes noise while preserving edges."""
        return cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)

    @staticmethod
    def adaptive_threshold(img: np.ndarray) -> np.ndarray:
        """Adaptive Gaussian threshold for handling uneven lighting."""
        return cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
        )

    @staticmethod
    def otsu_threshold(img: np.ndarray) -> np.ndarray:
        """Otsu's binarization — good for bimodal histograms."""
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def deskew(img: np.ndarray) -> np.ndarray:
        """Correct rotation/skew by detecting text line angles."""
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 50:
            return img
        try:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if abs(angle) < 0.5:
                return img
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                  borderMode=cv2.BORDER_REPLICATE)
        except Exception:
            return img

    @staticmethod
    def sharpen(img: np.ndarray) -> np.ndarray:
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        return cv2.filter2D(img, -1, kernel)

    @staticmethod
    def morphological_cleanup(img: np.ndarray) -> np.ndarray:
        """Close small gaps in characters, remove tiny noise specks."""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        return img

    @staticmethod
    def remove_borders(img: np.ndarray) -> np.ndarray:
        """Remove dark borders/scan artifacts."""
        contours, _ = cv2.findContours(cv2.bitwise_not(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        ih, iw = img.shape[:2]
        margin = 10
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(iw - x, w + 2 * margin)
        h = min(ih - y, h + 2 * margin)
        # Only crop if the detected region is at least 50% of the image
        if w * h > 0.5 * iw * ih:
            return img[y:y+h, x:x+w]
        return img

    def full_pipeline(self, img: np.ndarray, verbose: bool = False) -> list[np.ndarray]:
        """
        Return multiple preprocessed variants of the image.
        Running OCR on multiple variants and picking the best result
        dramatically improves accuracy.
        """
        variants = []

        # Variant 0: original (possibly upscaled)
        original = self.upscale(img.copy())
        variants.append(("original_upscaled", original if len(original.shape) == 2
                         else self.to_grayscale(original)))

        gray = self.to_grayscale(self.upscale(img.copy()))

        # Variant 1: denoised + adaptive threshold
        denoised = self.denoise(gray)
        v1 = self.adaptive_threshold(denoised)
        v1 = self.deskew(v1)
        variants.append(("denoise_adaptive", v1))

        # Variant 2: denoised + Otsu threshold
        v2 = self.otsu_threshold(denoised)
        v2 = self.deskew(v2)
        variants.append(("denoise_otsu", v2))

        # Variant 3: sharpen + adaptive threshold
        sharpened = self.sharpen(gray)
        v3 = self.adaptive_threshold(sharpened)
        variants.append(("sharpen_adaptive", v3))

        # Variant 4: full pipeline (denoise + sharpen + morph + deskew)
        full = self.denoise(gray)
        full = self.sharpen(full)
        full = self.otsu_threshold(full)
        full = self.morphological_cleanup(full)
        full = self.deskew(full)
        variants.append(("full_pipeline", full))

        # Variant 5: high-contrast
        pil_img = Image.fromarray(gray)
        enhancer = ImageEnhance.Contrast(pil_img)
        high_contrast = np.array(enhancer.enhance(2.0))
        high_contrast = self.otsu_threshold(high_contrast)
        variants.append(("high_contrast", high_contrast))

        if verbose:
            print(f"  [preprocess] Generated {len(variants)} image variants")

        return variants


# ---------------------------------------------------------------------------
# OCR Engines
# ---------------------------------------------------------------------------

class TesseractEngine:
    """Tesseract OCR with multiple page segmentation modes."""

    def __init__(self, lang: str = "eng"):
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract is not installed. Install with: sudo apt-get install tesseract-ocr")
        import pytesseract
        self.pytesseract = pytesseract
        self.lang = lang

    def ocr(self, img: np.ndarray, psm: int = 3) -> OCRResult:
        """Run Tesseract with a specific PSM mode."""
        config = f"--oem 3 --psm {psm}"
        try:
            data = self.pytesseract.image_to_data(
                img, lang=self.lang, config=config, output_type=self.pytesseract.Output.DICT
            )
            lines = {}
            for i, text in enumerate(data["text"]):
                if text.strip():
                    block = data["block_num"][i]
                    line = data["line_num"][i]
                    key = (block, line)
                    if key not in lines:
                        lines[key] = {"texts": [], "confs": []}
                    lines[key]["texts"].append(text)
                    conf = float(data["conf"][i])
                    if conf > 0:
                        lines[key]["confs"].append(conf)

            full_text_parts = []
            confidences = []
            details = []
            for key in sorted(lines.keys()):
                line_text = " ".join(lines[key]["texts"])
                line_confs = lines[key]["confs"]
                avg_conf = sum(line_confs) / len(line_confs) if line_confs else 0
                full_text_parts.append(line_text)
                confidences.append(avg_conf)
                details.append({"text": line_text, "confidence": avg_conf})

            text = "\n".join(full_text_parts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return OCRResult(text=text, confidence=avg_confidence,
                             engine=f"tesseract_psm{psm}", details=details)
        except Exception as e:
            return OCRResult(text="", confidence=0, engine=f"tesseract_psm{psm}",
                             details=[{"error": str(e)}])

    def multi_psm_ocr(self, img: np.ndarray) -> list[OCRResult]:
        """Run Tesseract with several PSM modes and return all results."""
        results = []
        for psm in [3, 4, 6, 11, 12]:
            result = self.ocr(img, psm=psm)
            if result.text.strip():
                results.append(result)
        return results


class EasyOCREngine:
    """EasyOCR — deep learning based OCR."""

    _reader = None
    _lang = None

    @classmethod
    def get_reader(cls, lang: str = "en"):
        if cls._reader is None or cls._lang != lang:
            import easyocr
            cls._reader = easyocr.Reader([lang], gpu=False, verbose=False)
            cls._lang = lang
        return cls._reader

    def __init__(self, lang: str = "en"):
        self.lang = lang

    def ocr(self, img: np.ndarray) -> OCRResult:
        try:
            reader = self.get_reader(self.lang)
            results = reader.readtext(img, detail=1, paragraph=False)

            # Sort by vertical position then horizontal
            results.sort(key=lambda r: (min(p[1] for p in r[0]), min(p[0] for p in r[0])))

            # Group into lines based on y-coordinate proximity
            lines = []
            current_line = []
            last_y = None
            for bbox, text, conf in results:
                y = min(p[1] for p in bbox)
                h = max(p[1] for p in bbox) - y
                threshold = max(h * 0.5, 10)
                if last_y is not None and abs(y - last_y) > threshold:
                    if current_line:
                        lines.append(current_line)
                    current_line = []
                current_line.append((bbox, text, conf))
                last_y = y
            if current_line:
                lines.append(current_line)

            text_lines = []
            confidences = []
            details = []
            for line in lines:
                # Sort words within a line by x-coordinate
                line.sort(key=lambda r: min(p[0] for p in r[0]))
                line_text = " ".join(w[1] for w in line)
                line_conf = sum(w[2] for w in line) / len(line) * 100
                text_lines.append(line_text)
                confidences.append(line_conf)
                details.append({"text": line_text, "confidence": line_conf})

            text = "\n".join(text_lines)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0

            return OCRResult(text=text, confidence=avg_conf, engine="easyocr", details=details)
        except Exception as e:
            return OCRResult(text="", confidence=0, engine="easyocr",
                             details=[{"error": str(e)}])


# ---------------------------------------------------------------------------
# File handling (PDF + images)
# ---------------------------------------------------------------------------

def load_images_from_file(file_path: str, pages: Optional[str] = None) -> list[np.ndarray]:
    """Load image(s) from a file. PDFs are converted to images page by page."""
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        from pdf2image import convert_from_path
        kwargs = {"dpi": 300}
        if pages:
            page_list = _parse_pages(pages)
            kwargs["first_page"] = min(page_list)
            kwargs["last_page"] = max(page_list)
        pil_images = convert_from_path(file_path, **kwargs)
        return [np.array(img.convert("RGB")) for img in pil_images]

    # Standard image formats
    pil_img = Image.open(file_path)

    # Handle multi-frame images (GIF, TIFF)
    frames = []
    try:
        while True:
            frames.append(np.array(pil_img.convert("RGB")))
            pil_img.seek(pil_img.tell() + 1)
    except EOFError:
        pass

    if not frames:
        frames = [np.array(pil_img.convert("RGB"))]

    return frames


def _parse_pages(pages_str: str) -> list[int]:
    """Parse page spec like '1-3,5,7-9' into a list of page numbers."""
    result = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(part))
    return result


# ---------------------------------------------------------------------------
# Result selection — pick the best OCR output
# ---------------------------------------------------------------------------

def pick_best_result(results: list[OCRResult], verbose: bool = False) -> OCRResult:
    """
    Select the best OCR result using a scoring heuristic that considers:
    - Confidence score from the engine
    - Text length (longer usually means more was detected)
    - Dictionary word ratio (heuristic for real text vs garbage)
    - Character quality (ratio of alphanumeric + common punctuation)
    """
    if not results:
        return OCRResult(text="", confidence=0, engine="none")

    scored = []
    for r in results:
        if not r.text.strip():
            continue
        text = r.text.strip()

        # Score components
        conf_score = r.confidence / 100.0  # normalize to 0-1

        # Character quality: ratio of "normal" characters
        normal_chars = sum(1 for c in text if c.isalnum() or c in " .,;:!?'-\"\n\t()[]{}/@#$%&*+=<>")
        char_quality = normal_chars / len(text) if text else 0

        # Text length bonus (longer = found more text, usually better)
        length_score = min(len(text) / 500, 1.0)

        # Word-like token ratio
        tokens = text.split()
        word_like = sum(1 for t in tokens if len(t) >= 2 and any(c.isalpha() for c in t))
        word_ratio = word_like / len(tokens) if tokens else 0

        # Combined score
        score = (conf_score * 0.4 + char_quality * 0.25 +
                 length_score * 0.15 + word_ratio * 0.2)

        if verbose:
            print(f"  [{r.engine}] conf={r.confidence:.1f} chars={len(text)} "
                  f"quality={char_quality:.2f} score={score:.3f}")

        scored.append((score, r))

    if not scored:
        return results[0] if results else OCRResult(text="", confidence=0, engine="none")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


# ---------------------------------------------------------------------------
# Main OCR function
# ---------------------------------------------------------------------------

def ocr_file(
    file_path: str,
    engine: str = "both",
    lang: str = "en",
    preprocess: Optional[bool] = None,
    pages: Optional[str] = None,
    show_confidence: bool = False,
    verbose: bool = False,
) -> str:
    """
    Extract text from an image or PDF file.

    Args:
        file_path: Path to the input file
        engine: "tesseract", "easyocr", or "both"
        lang: Language code
        preprocess: None=auto, True=force, False=skip
        pages: PDF page spec (e.g., "1-3")
        show_confidence: Show per-line confidence
        verbose: Print detailed processing info

    Returns:
        Extracted text as a string.
    """
    file_path = str(Path(file_path).resolve())
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if verbose:
        print(f"Processing: {file_path}")

    images = load_images_from_file(file_path, pages=pages)
    if verbose:
        print(f"  Loaded {len(images)} page(s)/frame(s)")

    preprocessor = ImagePreprocessor()

    # Map language codes
    tess_lang = {"en": "eng", "fr": "fra", "de": "deu", "es": "spa",
                 "it": "ita", "pt": "por", "nl": "nld", "ja": "jpn",
                 "ko": "kor", "zh": "chi_sim"}.get(lang, lang)

    all_page_texts = []

    for page_idx, img in enumerate(images):
        if verbose and len(images) > 1:
            print(f"\n  --- Page {page_idx + 1} ---")

        all_results = []

        # Determine whether to preprocess
        do_preprocess = preprocess if preprocess is not None else True

        if do_preprocess:
            variants = preprocessor.full_pipeline(img, verbose=verbose)
        else:
            gray = preprocessor.to_grayscale(preprocessor.upscale(img))
            variants = [("original", gray)]

        use_tesseract = engine in ("tesseract", "both") and TESSERACT_AVAILABLE
        use_easyocr = engine in ("easyocr", "both")

        if engine == "tesseract" and not TESSERACT_AVAILABLE:
            print("Warning: Tesseract not installed, falling back to EasyOCR", file=sys.stderr)
            use_easyocr = True

        for variant_name, variant_img in variants:
            if use_tesseract:
                tess = TesseractEngine(lang=tess_lang)
                tess_results = tess.multi_psm_ocr(variant_img)
                for r in tess_results:
                    r.engine = f"{r.engine}_{variant_name}"
                all_results.extend(tess_results)

            if use_easyocr:
                easy = EasyOCREngine(lang=lang)
                result = easy.ocr(variant_img)
                result.engine = f"easyocr_{variant_name}"
                if result.text.strip():
                    all_results.append(result)

        best = pick_best_result(all_results, verbose=verbose)

        if verbose:
            print(f"  Best engine: {best.engine} (confidence: {best.confidence:.1f}%)")

        if show_confidence and best.details:
            lines = []
            for d in best.details:
                if "text" in d:
                    lines.append(f"[{d.get('confidence', 0):.1f}%] {d['text']}")
            all_page_texts.append("\n".join(lines))
        else:
            all_page_texts.append(best.text)

    return "\n\n--- Page Break ---\n\n".join(all_page_texts) if len(all_page_texts) > 1 else all_page_texts[0] if all_page_texts else ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="High-Accuracy OCR Tool — Extract text from any image or PDF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ocr_tool.py photo.jpg
  python ocr_tool.py scan.pdf --pages 1-5
  python ocr_tool.py receipt.png --engine easyocr --confidence
  python ocr_tool.py document.tiff --output result.txt --verbose
        """,
    )
    parser.add_argument("file", help="Path to image or PDF file")
    parser.add_argument("--engine", choices=["tesseract", "easyocr", "both"],
                        default="both", help="OCR engine (default: both)")
    parser.add_argument("--lang", default="en", help="Language code (default: en)")
    parser.add_argument("--preprocess", action="store_true", default=None,
                        help="Force preprocessing")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip preprocessing")
    parser.add_argument("--pages", default=None,
                        help="PDF pages to process, e.g. '1-3,5'")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    parser.add_argument("--confidence", action="store_true",
                        help="Show per-line confidence scores")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed processing info")

    args = parser.parse_args()

    preprocess = None
    if args.preprocess:
        preprocess = True
    elif args.no_preprocess:
        preprocess = False

    try:
        text = ocr_file(
            file_path=args.file,
            engine=args.engine,
            lang=args.lang,
            preprocess=preprocess,
            pages=args.pages,
            show_confidence=args.confidence,
            verbose=args.verbose,
        )

        if args.output:
            Path(args.output).write_text(text, encoding="utf-8")
            print(f"Output written to: {args.output}")
        else:
            print(text)

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
