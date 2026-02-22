#!/usr/bin/env python3
"""
Add a Book to the Reading Page
Searches for a book by name, fetches details, extracts cover colors,
and adds it to the reading page bookshelf.

Usage:
    python scripts/add_book.py "book name"

Example:
    python scripts/add_book.py "atomic habits"
    python scripts/add_book.py "the pragmatic programmer"

It will:
  1. Search Open Library for the book
  2. Let you pick the right edition
  3. Extract spine & text colors from the cover
  4. Ask you to write your summary/description
  5. Insert the book into _pages/reading.md
"""

import json
import os
import re
import sys
import tempfile
import textwrap
import urllib.request
import urllib.parse

try:
    from colorthief import ColorThief
except ImportError:
    print("colorthief not installed. Run: pip install colorthief")
    sys.exit(1)


# ─── Style examples for description generation ─────────────────────

STYLE_EXAMPLES = [
    {
        "book": "Sapiens: A Brief History of Humankind by Yuval Noah Harari",
        "desc": "yuval noah harari is this israeli historian who basically wrote the book on how we went from being just another ape to running the whole planet. the core idea that blew my mind is that humans dominate because we can cooperate in massive numbers through shared myths - things like money, nations, religions, corporations - none of these actually exist except in our collective imagination. he calls it the cognitive revolution, when we started telling stories and believing in things we can't see or touch. the agricultural revolution gets absolutely roasted - he calls it history's biggest fraud because we thought we domesticated wheat but wheat actually domesticated us, chaining us to backbreaking farm work and worse diets. he goes through empires, science, capitalism, all of it, showing how these imagined orders shape everything. some people say he oversimplifies stuff and cherry-picks evidence, which is fair, but honestly it completely changed how i think about why society works the way it does. like once you see that money is just a shared story we all believe in, you can't unsee it."
    },
    {
        "book": "Why We Sleep by Matthew Walker",
        "desc": "this book by matthew walker, a berkeley sleep scientist, completely changed how i think about sleep. it's not just some optional thing we do - it's super important for staying healthy and literally not dying early. if you sleep less than 6 hours, you're basically screwed with way higher risks of cancer and alzheimer's, plus your brain works like you're drunk. the book explains how the first half of sleep stores facts while the second half connects ideas and helps creativity. walker says we need 7-9 hours and should avoid screens before bed, keep rooms cool, stop caffeine after noon, and stay consistent with sleep times even on weekends. some people say he exaggerates stuff a bit, but honestly it made me realize sleep matters way more than i ever thought."
    },
    {
        "book": "Artificial Intelligence: A Guide for Thinking Humans by Melanie Mitchell",
        "desc": "melanie mitchell is this computer scientist who teaches at portland state and does research at the santa fe institute. in this book, she breaks down how ai actually works versus all the crazy hype we hear about it. she digs into machine learning, neural nets, deep learning and all that, but keeps coming back to this idea that current ai is just really good pattern matching without any real understanding. like it can do impressive stuff but has no clue what it's actually doing. she calls this the barrier of meaning - where ai can't really understand context or common sense or make the kind of connections humans easily do. she points out how brittle these systems are and how they can fail in super weird ways. the book was written before chatgpt but it's still super relevant and honestly makes complex stuff easy to get without dumbing it down too much."
    },
]


# ─── Color helpers (from get_book_colors.py) ────────────────────────

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])


def get_luminance(rgb):
    r, g, b = [x / 255.0 for x in rgb]
    r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def get_contrast_ratio(c1, c2):
    l1, l2 = get_luminance(c1), get_luminance(c2)
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def choose_text_color(spine_color, palette):
    best_color, best_contrast = None, 0
    for color in palette:
        contrast = get_contrast_ratio(spine_color, color)
        if contrast > best_contrast:
            best_contrast = contrast
            best_color = color
    if best_contrast < 3.0:
        lum = get_luminance(spine_color)
        return (26, 26, 26) if lum > 0.5 else (212, 165, 116)
    return best_color


def extract_colors(isbn):
    """Download cover and extract spine/text colors."""
    url = f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    print(f"  Fetching cover: {url}")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        try:
            urllib.request.urlretrieve(url, tmp.name)
        except Exception as e:
            print(f"  ⚠ Error downloading cover: {e}")
            return None, None
        try:
            ct = ColorThief(tmp.name)
            dominant = ct.get_color(quality=1)
            palette = ct.get_palette(color_count=6, quality=1)
        except Exception as e:
            print(f"  ⚠ Error extracting colors: {e}")
            return None, None
        finally:
            os.unlink(tmp.name)
    text_color = choose_text_color(dominant, palette)
    return rgb_to_hex(dominant), rgb_to_hex(text_color)


# ─── Open Library search ────────────────────────────────────────────

def search_books(query):
    """Search Open Library and return top results with ISBNs."""
    encoded = urllib.parse.quote(query)
    url = f"https://openlibrary.org/search.json?q={encoded}&limit=8&fields=title,author_name,isbn,first_publish_year,edition_count"
    print(f"\nSearching Open Library for \"{query}\"...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BookshelfScript/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"Error searching Open Library: {e}")
        return []

    results = []
    for doc in data.get("docs", []):
        isbns = doc.get("isbn", [])
        # prefer ISBN-13
        isbn13 = next((i for i in isbns if len(i) == 13), None)
        isbn = isbn13 or (isbns[0] if isbns else None)
        if not isbn:
            continue
        results.append({
            "title": doc.get("title", "Unknown"),
            "author": ", ".join(doc.get("author_name", ["Unknown"])),
            "isbn": isbn,
            "year": doc.get("first_publish_year", ""),
        })
    return results


# ─── Generate description via Claude ────────────────────────────────

def load_env_file():
    """Load API keys from .env file in repo root."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    env_path = os.path.join(repo_root, ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def generate_description(title, author):
    """Use Claude API to generate a book description in the user's style."""
    load_env_file()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  ⚠ ANTHROPIC_API_KEY not set. Add it to .env or export it.")
        return None

    examples_text = "\n\n".join(
        f"Book: {ex['book']}\nDescription: {ex['desc']}" for ex in STYLE_EXAMPLES
    )

    prompt = f"""Write a book description/summary for \"{title}\" by {author}.

Here are examples of how I write book descriptions. Match this EXACT style:
- all lowercase, no capitalization except names of specific concepts the author coined
- casual, conversational tone like talking to a friend
- starts by introducing the author briefly (who they are)
- explains the core ideas and key takeaways
- uses phrases like "honestly", "like", "basically", "super", "stuff"
- uses dashes for asides and parenthetical thoughts
- includes a balanced take (mentions criticism or caveats briefly)
- ends with a personal reflection on how the book changed your thinking
- one continuous paragraph, no line breaks
- no quotation marks around the description
- around 100-180 words

Examples:

{examples_text}

Now write one for \"{title}\" by {author}. Output ONLY the description text, nothing else."""

    body = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    print("  Generating description with Claude...")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        desc = data["content"][0]["text"].strip()
        # Clean up: remove surrounding quotes if model added them
        if desc.startswith('"') and desc.endswith('"'):
            desc = desc[1:-1]
        return desc
    except Exception as e:
        print(f"  ⚠ Error calling Claude API: {e}")
        return None


# ─── Inject into reading.md ─────────────────────────────────────────

def get_reading_md_path():
    """Resolve path to _pages/reading.md relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    return os.path.join(repo_root, "_pages", "reading.md")


def build_book_entry(title, author, isbn, spine_color, text_color, desc):
    """Build the JS object string for a book entry."""
    # Escape quotes in desc
    safe_desc = desc.replace('"', '\\"')
    safe_title = title.replace('"', '\\"')
    safe_author = author.replace('"', '\\"')
    entry = (
        f'    {{\n'
        f'      title: "{safe_title}",\n'
        f'      author: "{safe_author}",\n'
        f'      isbn: "{isbn}",\n'
        f'      spineColor: "{spine_color}",\n'
        f'      textColor: "{text_color}",\n'
        f'      desc: "{safe_desc}"\n'
        f'    }}'
    )
    return entry


def inject_book(entry_str):
    """Insert the new book entry at the end of the books array in reading.md."""
    md_path = get_reading_md_path()
    with open(md_path, "r") as f:
        content = f.read()

    # Find the last book object's closing brace before the array close "];"
    # Pattern: last "}" before the "];" that closes the books array
    # We look for the pattern:  } followed by newline(s) and ];
    pattern = r"(      desc: \"[^\"]*\"\n    \})\n  \];"
    match = re.search(pattern, content)
    if not match:
        # Fallback: find "];" after "var books = ["
        pattern2 = r"(\})\s*\n\s*\];"
        matches = list(re.finditer(pattern2, content))
        if not matches:
            print("⚠ Could not find the books array in reading.md!")
            print("  Printing the entry so you can paste it manually:\n")
            print(entry_str)
            return False
        match = matches[-1]

    insert_pos = match.end(1)
    new_content = content[:insert_pos] + ",\n" + entry_str + content[insert_pos:]

    with open(md_path, "w") as f:
        f.write(new_content)

    return True


# ─── Main flow ──────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/add_book.py \"book name\"")
        print("Example: python scripts/add_book.py \"atomic habits\"")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    results = search_books(query)

    if not results:
        print("No results found. Try a different search term.")
        sys.exit(1)

    # Show results
    print(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results):
        year_str = f" ({r['year']})" if r['year'] else ""
        print(f"  [{i + 1}] {r['title']}{year_str}")
        print(f"      by {r['author']}  |  ISBN: {r['isbn']}")
        print()

    # Pick one
    while True:
        choice = input("Pick a number (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            sys.exit(0)
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(results):
                break
        except ValueError:
            pass
        print("Invalid choice, try again.")

    book = results[idx]
    print(f"\n✓ Selected: {book['title']} by {book['author']}")

    # Extract colors
    print("\nExtracting cover colors...")
    spine_color, text_color = extract_colors(book["isbn"])
    if not spine_color:
        print("Could not extract colors. You can set them manually.")
        spine_color = input("  spineColor (hex, e.g. #eae8e0): ").strip() or "#eae8e0"
        text_color = input("  textColor  (hex, e.g. #1a1a1a): ").strip() or "#1a1a1a"

    print(f"  spineColor: {spine_color}")
    print(f"  textColor:  {text_color}")

    # Generate description
    print("\nGenerating book description in your style...")
    desc = generate_description(book["title"], book["author"])

    if desc:
        print("\n" + "=" * 60)
        print("Generated description:\n")
        print(textwrap.fill(desc, width=80))
        print("\n" + "=" * 60)

        while True:
            action = input("\n[a]ccept / [r]egenerate / [e]dit manually / [q]uit: ").strip().lower()
            if action == "a":
                break
            elif action == "r":
                print("\nRegenerating...")
                desc = generate_description(book["title"], book["author"])
                if desc:
                    print("\n" + "=" * 60)
                    print("Generated description:\n")
                    print(textwrap.fill(desc, width=80))
                    print("\n" + "=" * 60)
                else:
                    print("Generation failed. Falling back to manual input.")
                    action = "e"
            if action == "e":
                print("\nType your description (press Enter twice when done):\n")
                lines = []
                while True:
                    line = input()
                    if line == "" and lines:
                        break
                    lines.append(line)
                desc = " ".join(lines).strip()
                break
            elif action == "q":
                sys.exit(0)
    else:
        print("\nAuto-generation unavailable. Write it yourself:")
        print("(Type your description, press Enter twice when done)\n")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        desc = " ".join(lines).strip()

    if not desc:
        print("No description provided. Aborting.")
        sys.exit(1)

    # Build entry
    entry = build_book_entry(book["title"], book["author"], book["isbn"],
                             spine_color, text_color, desc)

    print("\n" + "=" * 60)
    print("Here's the book entry:\n")
    print(entry)
    print("\n" + "=" * 60)

    # Confirm and inject
    confirm = input("\nAdd this to reading.md? (y/n): ").strip().lower()
    if confirm != "y":
        print("Cancelled. Entry printed above if you want to paste it manually.")
        sys.exit(0)

    if inject_book(entry):
        print("\n✓ Book added to _pages/reading.md!")
        print("  Run your site locally to verify it looks good.")
    else:
        print("\n⚠ Could not auto-insert. Paste the entry above into reading.md manually.")


if __name__ == "__main__":
    main()
