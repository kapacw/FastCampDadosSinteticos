"""
SinistroScan: treinamento do detector YOLOv8.

Treina um YOLOv8 (nano por padrão) usando exclusivamente o dataset sintético
gerado no Blender e avalia no split de teste ao final.

Nota: os dados de treino são 100% sintéticos; os pesos iniciais são os
pré-treinados no COCO (transfer learning, prática padrão) — o Ultralytics
baixa yolov8n.pt (~6 MB) automaticamente na primeira execução.

Uso (python do venv):
    python treinamento/treinar.py --data C:/caminho/dataset/data.yaml \
        [--modelo yolov8n.pt --epocas 100 --batch 32]
"""

import argparse
import os

from ultralytics import YOLO

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(RAIZ, "dataset", "data.yaml"))
    p.add_argument("--modelo", default="yolov8n.pt")
    p.add_argument("--epocas", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--nome", default="sinistroscan")
    args = p.parse_args()

    modelo = YOLO(args.modelo)
    modelo.train(
        data=args.data,
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=42,
        patience=30,
        project=os.path.join(RAIZ, "runs"),
        name=args.nome,
        exist_ok=True,
    )

    metricas = modelo.val(data=args.data, split="test",
                          project=os.path.join(RAIZ, "runs"),
                          name=args.nome + "_teste", exist_ok=True)
    print("\n===== RESULTADOS NO TESTE =====")
    print(f"mAP50:     {metricas.box.map50:.4f}")
    print(f"mAP50-95:  {metricas.box.map:.4f}")
    for i, ap in enumerate(metricas.box.maps):
        print(f"  classe {i}: mAP50-95 {ap:.4f}")


if __name__ == "__main__":
    main()
