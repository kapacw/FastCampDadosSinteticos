"""
SinistroScan: Demo de triagem de sinistro.

Aplica o detector treinado a uma foto (ou pasta de fotos) e produz a
pré-análise: peças detectadas, região provável do impacto e a imagem anotada.

Uso (python do venv):
    python scripts/triagem_demo.py --imagem foto.jpg \
        [--modelo modelos/best.pt --conf 0.50 --saida docs/img/triagem]

O caminho da imagem anotada é gravado no JSON de forma relativa à raiz do
projeto, para que o resultado possa ser publicado sem expor caminhos locais.
"""

import argparse
import json
import os

import cv2
from ultralytics import YOLO

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

LIMIAR_REGIAO = 1.0

REGIAO_POR_CLASSE = {
    "parachoque_dianteiro": "frontal",
    "capo": "frontal",
    "farol": "frontal",
    "porta": "lateral",
    "retrovisor": "lateral",
    "parachoque_traseiro": "traseira",
    "lanterna": "traseira",
}


def triagem(modelo, caminho, conf, dir_saida):
    resultado = modelo.predict(caminho, conf=conf, verbose=False)[0]
    nomes = resultado.names

    pecas = []
    pontuacao = {"frontal": 0.0, "lateral": 0.0, "traseira": 0.0}
    for caixa in resultado.boxes:
        classe = nomes[int(caixa.cls)]
        confianca = float(caixa.conf)
        pecas.append({
            "classe": classe,
            "confianca": round(confianca, 3),
            "bbox_xyxy": [round(v, 1) for v in caixa.xyxy[0].tolist()],
        })
        regiao = REGIAO_POR_CLASSE.get(classe)
        if regiao:
            pontuacao[regiao] += confianca

    melhor = max(pontuacao, key=pontuacao.get) if pontuacao else None
    regiao_provavel = (melhor if melhor and pontuacao[melhor] >= LIMIAR_REGIAO
                       else "indefinida")

    os.makedirs(dir_saida, exist_ok=True)
    nome = os.path.splitext(os.path.basename(caminho))[0]
    caminho_img = os.path.join(dir_saida, f"triagem_{nome}.png")
    cv2.imwrite(caminho_img, resultado.plot())

    try:
        caminho_img_rel = os.path.relpath(caminho_img, RAIZ)
        if caminho_img_rel.startswith(".."):
            caminho_img_rel = os.path.basename(caminho_img)
    except ValueError:
        caminho_img_rel = os.path.basename(caminho_img)
    caminho_img_rel = caminho_img_rel.replace("\\", "/")

    return {
        "arquivo": os.path.basename(caminho),
        "regiao_provavel_do_impacto": regiao_provavel,
        "pontuacao_por_regiao": {k: round(v, 3) for k, v in pontuacao.items()},
        "pecas_detectadas": sorted(pecas, key=lambda p: -p["confianca"]),
        "pre_lista_para_orcamento": sorted({p["classe"] for p in pecas}),
        "imagem_anotada": caminho_img_rel,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--imagem", required=True, help="foto ou pasta de fotos")
    p.add_argument("--modelo", default=os.path.join(RAIZ, "modelos", "best.pt"))
    p.add_argument("--conf", type=float, default=0.50)
    p.add_argument("--saida", default=os.path.join(RAIZ, "docs", "img", "triagem"))
    args = p.parse_args()

    modelo = YOLO(args.modelo)
    if os.path.isdir(args.imagem):
        arquivos = [os.path.join(args.imagem, a) for a in sorted(os.listdir(args.imagem))
                    if a.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    else:
        arquivos = [args.imagem]

    laudos = [triagem(modelo, a, args.conf, args.saida) for a in arquivos]
    print(json.dumps(laudos, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
