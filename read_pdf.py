from pathlib import Path

try:
    import pypdf
except ImportError:
    print("Error: pypdf module is not installed")
    raise SystemExit(1)


def extract_pdf_text(pdf_path: Path) -> str:
    reader = pypdf.PdfReader(str(pdf_path))
    return "".join(page.extract_text() or "" for page in reader.pages)


def main() -> None:
    candidates = sorted(Path(".").glob("*.pdf"))
    if not candidates:
        print("Error: no PDF files found in current directory")
        return

    target = candidates[0]
    print(f"Reading: {target}")
    print(extract_pdf_text(target))


if __name__ == "__main__":
    main()
