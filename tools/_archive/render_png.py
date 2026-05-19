import fitz
import os
import sys

# Parse arguments
if len(sys.argv) < 2:
    print("Render PDF to PNG")
    print("\nUsage:")
    print("  python render_png.py <pdf_path> [dpi]")
    print("\nExamples:")
    print("  python render_png.py file.pdf")
    print("  python render_png.py file.pdf 72   # no scale (original size)")
    print("  python render_png.py file.pdf 300  # enlarged render")
    print()
    print("Default DPI: 72 (no scale, original PDF display size)")
    sys.exit(1)

pdf_path = sys.argv[1]
dpi = int(sys.argv[2]) if len(sys.argv) > 2 else 300

doc = fitz.open(pdf_path)

print(f"PDF: {os.path.basename(pdf_path)}")
print(f"Total pages: {len(doc)}")
print(f"DPI: {dpi}")
print("="*60)

# Create output directory based on PDF name
pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
output_dir = f"rendered_{pdf_basename}_dpi{dpi}"
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")
print("="*60)

# Render all pages
zoom = dpi / 72  # Convert DPI to zoom factor
mat = fitz.Matrix(zoom, zoom)
print(f"Zoom factor: {zoom:.2f}x")

for page_num in range(len(doc)):
    page = doc[page_num]
    
    # Render page
    pix = page.get_pixmap(matrix=mat)
    
    # Save as PNG
    output_path = f"{output_dir}/page_{page_num}.png"
    pix.save(output_path)
    
    print(f"Page {page_num}: {pix.width}x{pix.height} → {output_path}")

print("="*60)
print(f"✅ Rendered {len(doc)} pages to {output_dir}/")