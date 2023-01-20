import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


def strhex_to_rgbvalues(strhex: str) -> tuple[int, int, int]:
    r = int(strhex[1:3], base=16)
    g = int(strhex[3:5], base=16)
    b = int(strhex[5:7], base=16)
    return r, g, b


def rgbvalues_to_strhex(rgb) -> str:
    r = f"{rgb[0]:02x}"
    g = f"{rgb[1]:02x}"
    b = f"{rgb[2]:02x}"
    return f"#{r}{g}{b}"


class BatchGen:
    def __init__(self, data_jsonl_path: Path, batch_size: int):
        self.batch_size = batch_size

        database = []
        self.use_bg_hl_groups = {
            "Normal",
            "NonText",
            "StatusLine",
            "LineNr",
            "CursorLineNr",
        }
        with open(data_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                colorscheme = json.loads(line.strip("\n"))
                normal_fg = colorscheme["colorscheme"]["Normal"]["guifg"]
                normal_bg = colorscheme["colorscheme"]["Normal"]["guibg"]
                if normal_bg == "":
                    normal_bg = "#000000"
                if normal_fg == "" and normal_fg == "":
                    continue

                row = {}
                for hl_group, fg_bg_pair in colorscheme["colorscheme"].items():
                    fg = fg_bg_pair["guifg"]
                    if fg == "":
                        fg = normal_fg
                    row[hl_group] = strhex_to_rgbvalues(fg)
                    if hl_group in self.use_bg_hl_groups:
                        bg = fg_bg_pair["guibg"]
                        if bg == "":
                            bg = normal_bg
                        row[f"{hl_group}_bg"] = strhex_to_rgbvalues(bg)
                database.append({"repo": colorscheme["repo_name"], "name": colorscheme["colorscheme_name"], **row})
        df = pd.DataFrame(database).set_index(["repo", "name"])
        self.df = df

        def gen():
            for row in df.itertuples(index=False):
                yield np.array(row) / 255.0

        self.xs = np.array(list(gen()))
        self.ds = (
            tf.data.Dataset.from_tensor_slices(self.xs)
            .shuffle(len(self.xs), reshuffle_each_iteration=True)
            .batch(batch_size)
            .prefetch(10)
        )
        self.dataset_mean = self.xs.mean(axis=0)
        self.dataset_std = self.xs.std(axis=0)
        self.input_length = self.xs.shape[1]
        self.datasize = self.xs.shape[0]
        self.hl_group_to_index = {str(c): i for i, c in enumerate(df.columns)}

    def visualize(self, inputs):
        def to256(x: np.ndarray) -> tuple[int, int, int]:
            return cast(tuple[int, int, int], tuple((x * 255).astype(np.uint8)))

        def get_color(inputs: np.ndarray, hl_group: str) -> tuple[int, int, int]:
            return to256(inputs[self.hl_group_to_index[hl_group]])

        canvases = []
        for i in range(len(inputs)):
            normal_bg = get_color(inputs[i], "Normal_bg")
            canvas = Image.fromarray(np.ones((400, 400, 3), dtype=np.uint8) * normal_bg)
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.truetype("DejaVuSansMono.ttf", 24)

            draw.polygon([(0, 0), (0, 364), (36, 364), (36, 0)], fill=get_color(inputs[i], "LineNr_bg"))
            for linenum, j in enumerate(range(4, 400, 36)):
                draw.text((4, j), f"{linenum:2}", fill=get_color(inputs[i], "LineNr"), font=font)
                if linenum >= 9:
                    break
            draw.polygon([(0, j), (0, j + 36), (400, j + 36), (400, j)], fill=get_color(inputs[i], "StatusLine_bg"))
            draw.text((8, j + 4), "hello_world.py", fill=get_color(inputs[i], "StatusLine"), font=font)
            draw.text((50, 4), "@dataclass", fill=get_color(inputs[i], "Structure"), font=font)
            draw.text((50, 40), "class", fill=get_color(inputs[i], "Keyword"), font=font)
            draw.text((140, 40), "Foo", fill=get_color(inputs[i], "Type"), font=font)
            draw.text((180, 40), "(", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((194, 40), "ABC", fill=get_color(inputs[i], "Type"), font=font)
            draw.text((234, 40), "):", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((80, 76), "def", fill=get_color(inputs[i], "Keyword"), font=font)
            draw.text((140, 76), "__init__", fill=get_color(inputs[i], "Special"), font=font)
            draw.text((260, 76), "(", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((106, 112), "self", fill=get_color(inputs[i], "Special"), font=font)
            draw.text((160, 112), ",", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((106, 148), "name", fill=get_color(inputs[i], "Normal"), font=font)
            draw.text((160, 148), ",", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((80, 184), "):", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((106, 216), "hello", fill=get_color(inputs[i], "Function"), font=font)
            draw.text((176, 216), "(", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((186, 216), '"hello!"', fill=get_color(inputs[i], "String"), font=font)
            draw.text((306, 216), ",", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((180, 252), "1", fill=get_color(inputs[i], "Number"), font=font)
            draw.text((210, 252), ",", fill=get_color(inputs[i], "Delimiter"), font=font)
            draw.text((186, 288), "True", fill=get_color(inputs[i], "Boolean"), font=font)
            draw.text((250, 288), ")", fill=get_color(inputs[i], "Delimiter"), font=font)

            canvases.append(canvas)

        return np.stack(canvases, axis=0)

    def to_vim(self, inputs, output_dir: Path):
        for i in range(len(inputs)):
            lines = [
                "hi clear",
                "if exists('syntax_on')",
                "  syntax reset",
                "endif",
                f"let g:colors_name = 'gen_{i:02}'",
                "",
            ]
            for hl_group, index in self.hl_group_to_index.items():
                if hl_group.endswith("_bg"):
                    continue
                if hl_group in self.use_bg_hl_groups:
                    bg_index = self.hl_group_to_index[f"{hl_group}_bg"]
                    guibg = " guibg=" + rgbvalues_to_strhex(inputs[i, bg_index])
                guifg = f"guifg={rgbvalues_to_strhex(inputs[i, index])}"
                lines.append(f"hi {hl_group} {guifg} {guibg}")
            with open(output_dir / f"gen_{i:02}.vim", "w", encoding="utf-8") as f:
                print("\n".join(lines), file=f)
