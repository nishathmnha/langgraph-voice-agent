from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import shutil
import subprocess
import textwrap


ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
DIAGRAMS_DIR = ROOT / "diagrams"
PDF_OUT = ROOT / "exports" / "pdf"
PNG_OUT = ROOT / "exports" / "png"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/consolab.ttf" if bold else "C:/Windows/Fonts/consola.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


TITLE_FONT = load_font(28, True)
H1_FONT = load_font(22, True)
H2_FONT = load_font(17, True)
BODY_FONT = load_font(12)
CODE_FONT = load_font(11)


def line_height(font: ImageFont.ImageFont) -> int:
    box = font.getbbox("Ag")
    return box[3] - box[1] + 6


def wrap_line(text: str, max_chars: int) -> list[str]:
    if not text:
        return [""]
    stripped = text.lstrip()
    if stripped.startswith("- "):
        wrapped = textwrap.wrap(stripped[2:], width=max_chars - 2)
        return [f"- {line}" if i == 0 else f"  {line}" for i, line in enumerate(wrapped)]
    return textwrap.wrap(text, width=max_chars) or [""]


def md_to_pages(title: str, content: str) -> list[Image.Image]:
    width, height = 1240, 1754
    margin = 90
    max_chars = 112
    pages: list[Image.Image] = []
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = margin

    def new_page():
        nonlocal image, draw, y
        pages.append(image)
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        y = margin

    draw.text((margin, y), title, fill="#111111", font=TITLE_FONT)
    y += line_height(TITLE_FONT) + 18

    in_code = False
    for raw in content.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            y += 4
            continue

        if line.startswith("# "):
            font = H1_FONT
            text = line[2:]
            spacing = 10
        elif line.startswith("## "):
            font = H2_FONT
            text = line[3:]
            spacing = 8
        elif line.startswith("|") or in_code:
            font = CODE_FONT
            text = line
            spacing = 2
        else:
            font = BODY_FONT
            text = line
            spacing = 4

        for wrapped in wrap_line(text, max_chars):
            if y > height - margin:
                new_page()
            draw.text((margin, y), wrapped, fill="#222222", font=font)
            y += line_height(font)
        y += spacing

    pages.append(image)
    return pages


def export_pdf(path: Path, output: Path):
    title = path.stem.replace("_", " ").title()
    pages = md_to_pages(title, path.read_text(encoding="utf-8"))
    first, rest = pages[0], pages[1:]
    first.save(output, save_all=True, append_images=rest)


def export_mermaid(diagram: Path, output: Path):
    mmdc = shutil.which("mmdc")
    if not mmdc:
        raise RuntimeError("Mermaid CLI (mmdc) was not found on PATH.")
    subprocess.run(
        [mmdc, "-i", str(diagram), "-o", str(output), "-b", "white"],
        check=True,
    )


def main():
    PDF_OUT.mkdir(parents=True, exist_ok=True)
    PNG_OUT.mkdir(parents=True, exist_ok=True)

    docs = sorted(DOCS_DIR.glob("*.md"))
    diagrams = sorted(DIAGRAMS_DIR.glob("*.mmd"))

    for doc in docs:
        export_pdf(doc, PDF_OUT / f"{doc.stem}.pdf")

    for diagram in diagrams:
        export_mermaid(diagram, PNG_OUT / f"{diagram.stem}.png")
        export_mermaid(diagram, PDF_OUT / f"{diagram.stem}.pdf")

    print(f"Exported {len(docs)} document PDFs to {PDF_OUT}")
    print(f"Exported {len(diagrams)} diagram PNGs to {PNG_OUT}")
    print(f"Exported {len(diagrams)} diagram PDFs to {PDF_OUT}")


if __name__ == "__main__":
    main()
