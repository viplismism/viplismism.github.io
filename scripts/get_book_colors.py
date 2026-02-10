#!/usr/bin/env python3
"""
Book Color Extractor
Extracts dominant colors from book covers for the reading page bookshelf.

Usage:
    python scripts/get_book_colors.py <ISBN>
    
Example:
    python scripts/get_book_colors.py 9780062316110
    
Output:
    spineColor: "#2c1810"
    textColor: "#d4a574"
"""

import sys
import tempfile
import urllib.request
from colorthief import ColorThief


def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color string."""
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def get_luminance(rgb):
    """Calculate relative luminance of a color."""
    r, g, b = [x / 255.0 for x in rgb]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def get_contrast_ratio(color1, color2):
    """Calculate contrast ratio between two colors."""
    lum1 = get_luminance(color1)
    lum2 = get_luminance(color2)
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    return (lighter + 0.05) / (darker + 0.05)


def choose_text_color(spine_color, palette):
    """Choose best text color from palette that contrasts well with spine."""
    best_color = None
    best_contrast = 0
    
    for color in palette:
        contrast = get_contrast_ratio(spine_color, color)
        if contrast > best_contrast:
            best_contrast = contrast
            best_color = color
    
    # If no good contrast found, use white or black
    if best_contrast < 3.0:
        spine_luminance = get_luminance(spine_color)
        if spine_luminance > 0.5:
            return (26, 26, 26)  # dark text on light spine
        else:
            return (212, 165, 116)  # warm light text on dark spine
    
    return best_color


def get_book_colors(isbn):
    """Fetch book cover and extract colors."""
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    
    print(f"Fetching cover from: {url}")
    
    # Download cover to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        try:
            urllib.request.urlretrieve(url, tmp.name)
        except Exception as e:
            print(f"Error downloading cover: {e}")
            return None, None
        
        # Extract colors
        try:
            color_thief = ColorThief(tmp.name)
            dominant = color_thief.get_color(quality=1)
            palette = color_thief.get_palette(color_count=6, quality=1)
        except Exception as e:
            print(f"Error extracting colors: {e}")
            return None, None
    
    # Spine color is the dominant color
    spine_color = dominant
    
    # Text color should contrast well with spine
    text_color = choose_text_color(spine_color, palette)
    
    return spine_color, text_color


def main():
    if len(sys.argv) < 2:
        print("Usage: python get_book_colors.py <ISBN>")
        print("Example: python get_book_colors.py 9780062316110")
        sys.exit(1)
    
    isbn = sys.argv[1]
    spine_rgb, text_rgb = get_book_colors(isbn)
    
    if spine_rgb and text_rgb:
        spine_hex = rgb_to_hex(spine_rgb)
        text_hex = rgb_to_hex(text_rgb)
        
        print("\n" + "=" * 40)
        print(f"ISBN: {isbn}")
        print("=" * 40)
        print(f'spineColor: "{spine_hex}",')
        print(f'textColor: "{text_hex}",')
        print("=" * 40)
    else:
        print("Failed to extract colors")
        sys.exit(1)


if __name__ == "__main__":
    main()
