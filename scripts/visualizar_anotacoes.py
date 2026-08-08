"""
SinistroScan: Conferência visual das anotações automáticas.

Desenha as bounding boxes YOLO sobre as imagens geradas e salva cópias
anotadas (+ um mosaico), para validar o pipeline e ilustrar README/relatório.

Uso (python do venv, fora do Blender):
    python scripts/visualizar_anotacoes.py --pool C:/caminho/dataset/pool \
        --n 12 --out docs/img/amostras
"""

import argparse
import os
import random

import cv2
import numpy as np

CLASSES = [
    "parachoque_dianteiro", "capo", "farol", "porta",
    "parachoque_traseiro", "lanterna", "retrovisor",
]
CORES = [
    (60, 200, 255), (80, 220, 100), (255, 200, 60), (250, 120, 250),
    (70, 110, 255), (90, 90, 235), (240, 240, 120),
]


def desenhar(caminho_img, caminho_txt):
    img = cv2.imread(caminho_img)
    alt, larg = img.shape[:2]
    n = 0
    if os.path.exists(caminho_txt):
        with open(caminho_txt, encoding="utf-8") as f:
            for linha in f:
                partes = linha.split()
                if len(partes) != 5:
                    continue
                c = int(partes[0])
                cx, cy, w, h = map(float, partes[1:])
                x0 = int((cx - w / 2) * larg)
                y0 = int((cy - h / 2) * alt)
                x1 = int((cx + w / 2) * larg)
                y1 = int((cy + h / 2) * alt)
                cor = CORES[c % len(CORES)]
                cv2.rectangle(img, (x0, y0), (x1, y1), cor, 2)
                rotulo = CLASSES[c] if c < len(CLASSES) else str(c)
                (tw, th), _ = cv2.getTextSize(rotulo, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                cv2.rectangle(img, (x0, y0 - th - 6), (x0 + tw + 4, y0), cor, -1)
                cv2.putText(img, rotulo, (x0 + 2, y0 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (10, 10, 10), 1, cv2.LINE_AA)
                n += 1
    return img, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", required=True, help="pasta com images/ e labels/")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--out", default=os.path.join("docs", "img", "amostras"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    dir_img = os.path.join(args.pool, "images")
    dir_lbl = os.path.join(args.pool, "labels")
    arquivos = sorted(a for a in os.listdir(dir_img) if a.endswith(".png"))
    rng = random.Random(args.seed)
    escolha = arquivos if len(arquivos) <= args.n else rng.sample(arquivos, args.n)
    escolha.sort()

    os.makedirs(args.out, exist_ok=True)
    paineis = []
    total = 0
    for nome in escolha:
        img, n = desenhar(os.path.join(dir_img, nome),
                          os.path.join(dir_lbl, nome.replace(".png", ".txt")))
        total += n
        cv2.imwrite(os.path.join(args.out, nome), img)
        paineis.append(cv2.resize(img, (320, 320)))

    if paineis:
        cols = 4
        while len(paineis) % cols:
            paineis.append(np.zeros_like(paineis[0]))
        linhas = [np.hstack(paineis[i:i + cols]) for i in range(0, len(paineis), cols)]
        mosaico = np.vstack(linhas)
        cv2.imwrite(os.path.join(args.out, "..", "amostras_grid.png"), mosaico)

    print(f"{len(escolha)} imagens anotadas ({total} caixas) salvas em {args.out}")


if __name__ == "__main__":
    main()
