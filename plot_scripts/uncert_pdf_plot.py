import os
from fpdf import FPDF

base_folder = "output"
variants = ["beforeuncert", "beforeuncert0.1", "beforeuncert0.001", "beforeuncert0.005"]
image_ids = ["000", "010", "020"]

pdf = FPDF(orientation="P", unit="mm", format="A4")
pdf.set_auto_page_break(auto=True, margin=10)

for image_id in image_ids:
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, f"Comparison for Image ID {image_id}", ln=True, align="C")

    y_offset = 20
    for variant in variants:
        folder_path = os.path.join(base_folder, variant, "test")
        matched_file = next(
            (f for f in os.listdir(folder_path) if f.startswith(f"rgb_{image_id}_") and f.endswith(".png")),
            None
        )
        if matched_file:
            image_path = os.path.join(folder_path, matched_file)
            display_width_mm = 180
            pdf.image(image_path, x=15, y=y_offset, w=display_width_mm)
            y_offset += 60
            pdf.set_xy(10, y_offset)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"{variant} - {matched_file}", ln=True)
            y_offset += 10

pdf.output("comparison_vertical.pdf")
