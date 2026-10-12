import os
from PIL import Image, ImageDraw, ImageFont

def make_figure(frames, output_path, title_suffix=""):
    col_w, row_h = 720, 960  # original resolution
    scale = 0.5
    w = int(col_w * scale)
    h = int(row_h * scale)

    header_h = 85
    row_label_w = 120
    margin = 10

    canvas_w = row_label_w + 3 * w + 4 * margin
    canvas_h = header_h + len(frames) * (h + margin) + margin

    img = Image.new('RGB', (canvas_w, canvas_h), '#ffffff')
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 26)
        font_sub = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
        font_label1 = ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf', 20)
        font_label2 = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
    except:
        font_title = font_sub = font_label1 = font_label2 = ImageFont.load_default()

    headers = [
        ('YOLOE-26l-visual', '(Scene Visual Exemplars)'),
        ('SAM 3 Spatial Masks', '(Zero-Shot Text Prompts)'),
        ('Gated Mask Refinement', '(YOLOE Boxes + SAM 3)')
    ]

    for col, (h_title, h_sub) in enumerate(headers):
        x_center = row_label_w + margin + col * (w + margin) + w // 2
        bbox1 = draw.textbbox((0, 0), h_title, font=font_title)
        bbox2 = draw.textbbox((0, 0), h_sub, font=font_sub)
        draw.text((x_center - (bbox1[2]-bbox1[0])//2, 12), h_title, fill='#111827', font=font_title)
        draw.text((x_center - (bbox2[2]-bbox2[0])//2, 48), h_sub, fill='#4b5563', font=font_sub)

    row_info = {
        450: ('Frame 450', 'Beanbag & Sofa'),
        750: ('Frame 750', 'Table & Window'),
        1500: ('Frame 1500', 'Armchair & Window'),
        2000: ('Frame 2000', 'Door (Rejected)')
    }

    for row, frame in enumerate(frames):
        y = header_h + row * (h + margin)
        
        # Row label on left
        r_t1, r_t2 = row_info[frame]
        draw.text((12, y + h//2 - 18), r_t1, fill='#111827', font=font_label1)
        draw.text((12, y + h//2 + 4), r_t2, fill='#4b5563', font=font_label2)
        
        paths = [
            f'outputs/room_refined/compare_YOLOE-26l-visual_{frame:06d}.jpg',
            f'outputs/sam_trial/sam3_{frame:06d}.jpg',
            f'outputs/sam_trial/hybrid_{frame:06d}.jpg'
        ]
        
        for col, p in enumerate(paths):
            x = row_label_w + margin + col * (w + margin)
            cell_im = Image.open(p).convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
            img.paste(cell_im, (x, y))
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline='#cbd5e1', width=1)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path, dpi=(300, 300))
    print(f'Saved {output_path}: {img.size}')

if __name__ == '__main__':
    make_figure([450, 750, 1500], 'paper/figures/fig_sam_comparison_3row.png')
    make_figure([450, 750], 'paper/figures/fig_sam_comparison_2row.png')
    make_figure([450, 750, 1500, 2000], 'paper/figures/fig_sam_comparison_4row.png')
    # Default to 3-row for paper
    make_figure([450, 750, 1500], 'paper/figures/fig_sam_comparison.png')
