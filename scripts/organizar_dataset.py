"""
SinistroScan: Organização do dataset em treinar/validar/testar.

Move as imagens/anotações de pool/ para a estrutura YOLO padrão e escreve o
data.yaml. Divisão embaralhada com semente fixa (reprodutível).

Uso:
    python scripts/organizar_dataset.py --dataset C:/caminho/dataset \
        [--treino 0.7 --valid 0.2 --seed 42]
"""

import argparse
import os
import random
import shutil

CLASSES = [
    "parachoque_dianteiro", "capo", "farol", "porta",
    "parachoque_traseiro", "lanterna", "retrovisor",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--treino", type=float, default=0.7)
    p.add_argument("--valid", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pool_img = os.path.join(args.dataset, "pool", "images")
    pool_lbl = os.path.join(args.dataset, "pool", "labels")
    arquivos = sorted(a for a in os.listdir(pool_img) if a.endswith(".png"))
    if not arquivos:
        raise SystemExit("pool vazio — rode gerar_dataset.py antes")

    rng = random.Random(args.seed)
    rng.shuffle(arquivos)
    n = len(arquivos)
    n_tr = int(n * args.treino)
    n_va = int(n * args.valid)
    splits = {
        "train": arquivos[:n_tr],
        "valid": arquivos[n_tr:n_tr + n_va],
        "test": arquivos[n_tr + n_va:],
    }

    for split, nomes in splits.items():
        d_img = os.path.join(args.dataset, split, "images")
        d_lbl = os.path.join(args.dataset, split, "labels")
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lbl, exist_ok=True)
        for nome in nomes:
            shutil.move(os.path.join(pool_img, nome), os.path.join(d_img, nome))
            txt = nome.replace(".png", ".txt")
            shutil.move(os.path.join(pool_lbl, txt), os.path.join(d_lbl, txt))
        print(f"{split}: {len(nomes)} imagens")

    meta = os.path.join(args.dataset, "pool", "metadata.csv")
    if os.path.exists(meta):
        shutil.move(meta, os.path.join(args.dataset, "metadata.csv"))
    shutil.rmtree(os.path.join(args.dataset, "pool"), ignore_errors=True)

    caminho_yaml = os.path.join(args.dataset, "data.yaml")
    with open(caminho_yaml, "w", encoding="utf-8") as f:
        f.write(f"path: {os.path.abspath(args.dataset)}\n")
        f.write("train: train/images\nval: valid/images\ntest: test/images\n")
        f.write("names:\n")
        for i, nome in enumerate(CLASSES):
            f.write(f"  {i}: {nome}\n")
    print("data.yaml escrito em", caminho_yaml)


if __name__ == "__main__":
    main()
