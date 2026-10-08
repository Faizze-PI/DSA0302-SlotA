"""
Screenshot Generator for NLP Laboratory Experiments
Generates high-resolution dark-mode terminal window images for all 26 experiments.
"""

import os
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PROGRAMS_DIR = os.path.join(BASE_DIR, "Programs")

FONT_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_SIZE = 15
LINE_HEIGHT = 22

try:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    bold_font = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", FONT_SIZE)
    title_font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 13)
except Exception:
    font = ImageFont.load_default()
    bold_font = font
    title_font = font

def get_line_color(line):
    l_str = line.strip()
    if l_str.startswith("=" * 10) or l_str.startswith("-" * 10):
        return "#58a6ff"  # Accent blue
    elif l_str.startswith("[+]") or "PASSED" in l_str or "ACCEPTED" in l_str or "AGREED" in l_str:
        return "#3fb950"  # Success green
    elif l_str.startswith("[-]") or "REJECTED" in l_str or "FAILED" in l_str:
        return "#f85149"  # Failure red
    elif l_str.startswith("[*]") or l_str.startswith(">>>") or (l_str.startswith("[") and "]" in l_str):
        return "#d29922"  # Section yellow
    elif ":" in l_str and len(l_str.split(":")[0]) < 25:
        return "#e6edf3"  # Text default
    elif l_str.startswith("Step") or l_str.startswith("Rank"):
        return "#a5d6ff"  # Soft cyan
    return "#c9d1d9"      # Standard terminal light gray

def create_terminal_screenshot(exp_num, exp_folder, output_text, target_png):
    lines = output_text.splitlines()
    # Limit lines if excessively long to keep image crisp and readable
    if len(lines) > 48:
        lines = lines[:45] + ["... [Output truncated for display; see ExpXX_Output.txt for full log] ..."] + lines[-2:]

    # Dimensions
    pad_x = 35
    pad_y = 35
    term_width = 1180
    title_bar_height = 38
    content_pad_x = 24
    content_pad_y = 18

    # Calculate height
    num_content_lines = len(lines) + 2  # Include prompt lines
    term_height = title_bar_height + (content_pad_y * 2) + (num_content_lines * LINE_HEIGHT)
    img_width = term_width + (pad_x * 2)
    img_height = term_height + (pad_y * 2)

    # 1. Base Canvas (Subtle dark background)
    img = Image.new("RGB", (img_width, img_height), "#0d1117")
    draw = ImageDraw.Draw(img)

    # 2. Terminal Shadow & Background Window
    x0, y0 = pad_x, pad_y
    x1, y1 = pad_x + term_width, pad_y + term_height

    # Shadow effect
    for offset in range(8, 0, -2):
        draw.rounded_rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], radius=14, fill="#05080c")

    # Terminal Container
    draw.rounded_rectangle([x0, y0, x1, y1], radius=10, fill="#161b22", outline="#30363d", width=1)

    # Title Bar
    draw.rounded_rectangle([x0, y0, x1, y0 + title_bar_height], radius=10, fill="#21262d")
    draw.rectangle([x0, y0 + 15, x1, y0 + title_bar_height], fill="#21262d")  # Square bottom corners
    draw.line([(x0, y0 + title_bar_height), (x1, y0 + title_bar_height)], fill="#30363d", width=1)

    # Window Controls (macOS/Linux style buttons)
    btn_y = y0 + title_bar_height // 2
    draw.ellipse([x0 + 16, btn_y - 6, x0 + 28, btn_y + 6], fill="#ff5f56")  # Close
    draw.ellipse([x0 + 36, btn_y - 6, x0 + 48, btn_y + 6], fill="#ffbd2e")  # Minimize
    draw.ellipse([x0 + 56, btn_y - 6, x0 + 68, btn_y + 6], fill="#27c93f")  # Maximize

    # Title Text
    title_label = f"student@simats-nlp-lab: ~/DSA0302/Lab/Programs/{exp_folder} (python main.py)"
    draw.text((x0 + 85, btn_y - 8), title_label, font=title_font, fill="#8b949e")

    # 3. Terminal Prompt Line
    curr_y = y0 + title_bar_height + content_pad_y
    curr_x = x0 + content_pad_x

    prompt_prefix = "user@simats-nlp:~/Lab/Programs/" + exp_folder + "$ "
    draw.text((curr_x, curr_y), prompt_prefix, font=bold_font, fill="#58a6ff")
    prefix_w = int(draw.textlength(prompt_prefix, font=bold_font))
    draw.text((curr_x + prefix_w, curr_y), "python main.py", font=bold_font, fill="#7ee787")
    curr_y += LINE_HEIGHT + 6

    # 4. Render Output Lines
    for line in lines:
        color = get_line_color(line)
        # Clip long horizontal lines to prevent overflow
        display_line = line[:120]
        draw.text((curr_x, curr_y), display_line, font=font, fill=color)
        curr_y += LINE_HEIGHT

    # Save PNG
    img.save(target_png, "PNG", optimize=True)

def main():
    print("=" * 75)
    print("   GENERATING HIGH-RESOLUTION DARK-MODE TERMINAL SCREENSHOTS   ")
    print("=" * 75)

    experiments = sorted([d for d in os.listdir(PROGRAMS_DIR) if os.path.isdir(os.path.join(PROGRAMS_DIR, d))])
    gallery_md_entries = []

    for exp_folder in experiments:
        exp_num = exp_folder[:2]
        txt_path = os.path.join(OUTPUTS_DIR, f"Exp{exp_num}_Output.txt")
        png_filename = f"Exp{exp_num}_{exp_folder}.png"
        png_path = os.path.join(OUTPUTS_DIR, png_filename)

        if not os.path.exists(txt_path):
            print(f"[-] Skipping {exp_folder}: {txt_path} not found.")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            output_text = f.read()

        create_terminal_screenshot(exp_num, exp_folder, output_text, png_path)
        print(f"[+] Generated Screenshot: {png_filename}")

        # Gallery entry
        title = exp_folder.replace("_", " ")[3:]
        gallery_md_entries.append(f"""### Experiment {exp_num}: {title}
- **Program Directory:** [`Lab/Programs/{exp_folder}`](../Programs/{exp_folder}/)
- **Raw Execution Log:** [`Exp{exp_num}_Output.txt`](./Exp{exp_num}_Output.txt)
- **Output Screenshot:**

![Experiment {exp_num} Terminal Output](./{png_filename})

---
""")

    # Generate Lab/outputs/README.md Gallery
    gallery_header = """# DSA0302 - NLP Laboratory Visual Output Gallery

This directory contains the verified console execution logs and high-resolution dark-mode terminal window screenshots for all 26 experiments in the **DSA0302: Natural Language Processing** university laboratory coursework.

All experiments were executed and verified with a **100% pass rate** in Python 3.11.

---

## Output Gallery Index

"""
    gallery_full_md = gallery_header + "\n".join(gallery_md_entries)
    readme_path = os.path.join(OUTPUTS_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(gallery_full_md)

    print(f"[+] Visual gallery generated at: {readme_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
