"""
SinistroScan: Geração de imagens sintéticas + anotação automática.

Para cada imagem:
  1. Randomiza cenário (frontal/lateral/traseira/geral), câmera, sol, exposição,
     cor da pintura, asfalto e distratores — com semente reprodutível por índice;
  2. Renderiza a imagem final (Cycles GPU, PNG);
  3. Renderiza um passe de IDs: material override com Object Info -> Object Index
     como emissão, salvo em EXR float. Cada instância de peça vira uma máscara
     exata (oclusões já resolvidas pelo próprio render);
  4. Deriva as bounding boxes YOLO da parte VISÍVEL de cada peça e grava o .txt;
  5. Registra os parâmetros da imagem em metadata.csv.

Uso (headless):
    blender -b blender/cena.blend -P scripts/gerar_dataset.py -- \
        --n 1200 --out C:/caminho/dataset --seed 42

Gera out/pool/images/img_XXXXX.png + out/pool/labels/img_XXXXX.txt.
A divisão train/valid/test é feita depois por scripts/organizar_dataset.py.
"""

import argparse
import colorsys
import csv
import json
import math
import os
import random
import sys
import time

import bpy
import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

MIN_PIXELS = 80
MIN_LADO = 4

CENARIOS = ("frontal", "lateral", "traseira", "geral")
PESOS_CENARIOS = (0.27, 0.30, 0.25, 0.18)


def achar_no(arvore, bl_idname):
    return next((n for n in arvore.nodes if n.bl_idname == bl_idname), None)


def bsdf_principled(mat):
    return achar_no(mat.node_tree, "ShaderNodeBsdfPrincipled")


def configurar_gpu(sc):
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for tipo in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = tipo
                break
            except Exception:
                continue
        prefs.get_devices()
        for d in prefs.devices:
            d.use = (d.type == prefs.compute_device_type)
        sc.cycles.device = "GPU"
        print("GPU:", prefs.compute_device_type)
    except Exception as e:
        print("AVISO: GPU indisponivel, usando CPU:", repr(e))


def material_ids():
    m = bpy.data.materials.new("__ids__")
    if m.node_tree is None:
        try:
            m.use_nodes = True
        except Exception:
            pass
    nt = m.node_tree
    nt.nodes.clear()
    info = nt.nodes.new("ShaderNodeObjectInfo")
    emissao = nt.nodes.new("ShaderNodeEmission")
    saida = nt.nodes.new("ShaderNodeOutputMaterial")
    try:
        comb = nt.nodes.new("ShaderNodeCombineColor")
        canais = ("Red", "Green", "Blue")
    except Exception:
        comb = nt.nodes.new("ShaderNodeCombineRGB")
        canais = ("R", "G", "B")
    for c in canais:
        nt.links.new(info.outputs["Object Index"], comb.inputs[c])
    nt.links.new(comb.outputs[0], emissao.inputs["Color"])
    emissao.inputs["Strength"].default_value = 1.0
    nt.links.new(emissao.outputs["Emission"], saida.inputs["Surface"])
    return m


def decodificar_ids(caminho_exr):
    img = bpy.data.images.load(caminho_exr)
    w, h = img.size
    buf = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(buf)
    ids = np.rint(buf[0::img.channels]).astype(np.int32).reshape(h, w)
    bpy.data.images.remove(img)
    return ids


def caixas_yolo(ids, mapa_classes):
    h, w = ids.shape
    linhas = []
    for iid in np.unique(ids):
        if iid <= 0 or iid not in mapa_classes:
            continue
        mask = ids == iid
        if int(mask.sum()) < MIN_PIXELS:
            continue
        tem_linha = np.any(mask, axis=1)
        tem_col = np.any(mask, axis=0)
        r0 = int(np.argmax(tem_linha))
        r1 = int(h - 1 - np.argmax(tem_linha[::-1]))
        c0 = int(np.argmax(tem_col))
        c1 = int(w - 1 - np.argmax(tem_col[::-1]))
        y0, y1 = h - 1 - r1, h - 1 - r0
        larg, alt = c1 - c0 + 1, y1 - y0 + 1
        if larg < MIN_LADO or alt < MIN_LADO:
            continue
        cx = (c0 + c1 + 1) / 2.0 / w
        cy = (y0 + y1 + 1) / 2.0 / h
        linhas.append((mapa_classes[iid], cx, cy, larg / w, alt / h))
    return linhas


def randomizar(sc, rng, refs):
    negativa = rng.random() < 0.10
    cenario = "negativa" if negativa else rng.choices(CENARIOS, weights=PESOS_CENARIOS)[0]

    refs["col_carro"].hide_render = negativa

    if cenario == "frontal":
        az = rng.uniform(-50, 50)
        dist = rng.uniform(1.8, 4.2)
        alvo = (rng.uniform(0.85, 1.45), rng.uniform(-0.5, 0.5), rng.uniform(0.35, 0.75))
    elif cenario == "traseira":
        az = rng.uniform(130, 230)
        dist = rng.uniform(1.8, 4.2)
        alvo = (rng.uniform(-1.45, -0.85), rng.uniform(-0.5, 0.5), rng.uniform(0.35, 0.75))
    elif cenario == "lateral":
        lado = rng.choice((1, -1))
        az = rng.uniform(55, 125) * lado
        dist = rng.uniform(2.0, 4.4)
        alvo = (rng.uniform(-0.7, 0.7), rng.uniform(-0.4, 0.4), rng.uniform(0.35, 0.75))
    elif cenario == "geral":
        az = rng.uniform(0, 360)
        dist = rng.uniform(4.0, 7.0)
        alvo = (rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.45, 0.75))
    else:
        az = rng.uniform(0, 360)
        dist = rng.uniform(3.0, 8.0)
        alvo = (rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0.3, 0.8))

    rad = math.radians(az)
    altura = rng.uniform(0.8, 1.9)
    refs["camera"].location = (dist * math.cos(rad), dist * math.sin(rad), altura)
    refs["alvo"].location = alvo
    lente = rng.uniform(24.0, 50.0)
    refs["camera"].data.lens = lente

    sol_elev = rng.uniform(5.0, 70.0)
    sol_rot = rng.uniform(0.0, 360.0)
    ceu = refs["ceu"]
    ceu.sun_elevation = math.radians(sol_elev)
    ceu.sun_rotation = math.radians(sol_rot)
    for prop, val in (("aerosol_density", rng.uniform(0.5, 3.0)),
                      ("air_density", rng.uniform(0.8, 1.2))):
        try:
            setattr(ceu, prop, val)
        except Exception:
            pass
    forca_ceu = rng.uniform(0.75, 1.25)
    refs["fundo"].inputs["Strength"].default_value = forca_ceu
    exposicao = rng.uniform(-6.0, -4.4)
    sc.view_settings.exposure = exposicao

    h_, s_, v_ = rng.random(), rng.uniform(0.25, 0.9), rng.uniform(0.12, 0.8)
    r, g, b = colorsys.hsv_to_rgb(h_, s_, v_)
    bsdf_principled(refs["pintura"]).inputs["Base Color"].default_value = (r, g, b, 1.0)

    tom = rng.uniform(0.05, 0.28)
    refs["asfalto_rampa"].color_ramp.elements[1].color = (tom, tom, tom * 1.04, 1.0)
    refs["asfalto_ruido"].inputs["Scale"].default_value = rng.uniform(30.0, 90.0)

    for d in refs["distratores"]:
        visivel = rng.random() < 0.5
        d.hide_render = not visivel
        if visivel:
            ang = rng.uniform(0, 2 * math.pi)
            raio = rng.uniform(6.0, 16.0)
            d.location = (raio * math.cos(ang), raio * math.sin(ang), 0.0)
            d.scale = (rng.uniform(0.4, 1.6), rng.uniform(0.4, 1.6), rng.uniform(0.3, 2.2))
            d.rotation_euler[2] = rng.uniform(0, math.pi)

    return {
        "cenario": cenario, "azimute": round(az, 1), "distancia": round(dist, 2),
        "altura_cam": round(altura, 2), "lente_mm": round(lente, 1),
        "sol_elevacao": round(sol_elev, 1), "sol_rotacao": round(sol_rot, 1),
        "forca_ceu": round(forca_ceu, 2), "exposicao": round(exposicao, 2),
        "pintura_rgb": "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)),
    }


def render_ids(sc, vl, refs, caminho_exr):
    est = {
        "samples": sc.cycles.samples,
        "denoise": sc.cycles.use_denoising,
        "adaptive": getattr(sc.cycles, "use_adaptive_sampling", None),
        "filtro": sc.render.filter_size,
        "formato": sc.render.image_settings.file_format,
        "modo_cor": sc.render.image_settings.color_mode,
        "forca": refs["fundo"].inputs["Strength"].default_value,
        "filepath": sc.render.filepath,
    }
    vl.material_override = refs["mat_ids"]
    sc.cycles.samples = 1
    sc.cycles.use_denoising = False
    try:
        sc.cycles.use_adaptive_sampling = False
    except Exception:
        pass
    sc.render.filter_size = 0.01
    sc.render.image_settings.file_format = "OPEN_EXR"
    sc.render.image_settings.color_mode = "RGB"
    try:
        sc.render.image_settings.color_depth = "32"
        sc.render.image_settings.exr_codec = "ZIP"
    except Exception:
        pass
    refs["fundo"].inputs["Strength"].default_value = 0.0
    sc.render.filepath = caminho_exr
    bpy.ops.render.render(write_still=True)

    vl.material_override = None
    sc.cycles.samples = est["samples"]
    sc.cycles.use_denoising = est["denoise"]
    if est["adaptive"] is not None:
        sc.cycles.use_adaptive_sampling = est["adaptive"]
    sc.render.filter_size = est["filtro"]
    sc.render.image_settings.file_format = est["formato"]
    sc.render.image_settings.color_mode = est["modo_cor"]
    refs["fundo"].inputs["Strength"].default_value = est["forca"]
    sc.render.filepath = est["filepath"]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1200)
    p.add_argument("--out", default=os.path.join(RAIZ, "dataset"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples", type=int, default=48)
    args = p.parse_args(argv)

    sc = bpy.context.scene
    vl = bpy.context.view_layer
    configurar_gpu(sc)
    sc.cycles.samples = args.samples

    classes = json.loads(sc["classes_json"])
    mapa_classes = {}
    objetos_classe = [o for o in sc.objects if o.type == "MESH" and "classe" in o.keys()]
    for o in objetos_classe:
        mapa_classes[o.pass_index] = classes.index(o["classe"])
    print(f"{len(objetos_classe)} objetos anotaveis, "
          f"{len(set(mapa_classes.values()))} classes, "
          f"{len(set(mapa_classes.keys()))} instancias")

    mundo_nt = sc.world.node_tree
    refs = {
        "camera": sc.objects["Camera"],
        "alvo": sc.objects["AlvoCamera"],
        "ceu": mundo_nt.nodes["Ceu"],
        "fundo": achar_no(mundo_nt, "ShaderNodeBackground"),
        "pintura": bpy.data.materials["Pintura"],
        "asfalto_rampa": bpy.data.materials["Asfalto"].node_tree.nodes["AsfaltoRampa"],
        "asfalto_ruido": bpy.data.materials["Asfalto"].node_tree.nodes["AsfaltoRuido"],
        "col_carro": bpy.data.collections["Carro"],
        "distratores": [o for o in sc.objects if o.name.startswith("distrator_")],
        "mat_ids": material_ids(),
    }

    dir_pool = os.path.join(args.out, "pool")
    dir_img = os.path.join(dir_pool, "images")
    dir_lbl = os.path.join(dir_pool, "labels")
    os.makedirs(dir_img, exist_ok=True)
    os.makedirs(dir_lbl, exist_ok=True)

    caminho_meta = os.path.join(dir_pool, "metadata.csv")
    meta_novo = not os.path.exists(caminho_meta)
    arq_meta = open(caminho_meta, "a", newline="", encoding="utf-8")
    campos = ["indice", "arquivo", "cenario", "azimute", "distancia", "altura_cam",
              "lente_mm", "sol_elevacao", "sol_rotacao", "forca_ceu", "exposicao",
              "pintura_rgb", "n_anotacoes"]
    escritor = csv.DictWriter(arq_meta, fieldnames=campos)
    if meta_novo:
        escritor.writeheader()

    exr_tmp = os.path.join(dir_pool, "_ids_tmp.exr")
    inicio = time.time()
    gerados = 0
    for i in range(args.n):
        nome = f"img_{i:05d}"
        caminho_png = os.path.join(dir_img, nome + ".png")
        caminho_txt = os.path.join(dir_lbl, nome + ".txt")
        if os.path.exists(caminho_png) and os.path.exists(caminho_txt):
            continue

        rng = random.Random(args.seed * 1_000_003 + i)
        registro = randomizar(sc, rng, refs)

        sc.render.filepath = caminho_png
        bpy.ops.render.render(write_still=True)

        render_ids(sc, vl, refs, exr_tmp)
        ids = decodificar_ids(exr_tmp)
        caixas = caixas_yolo(ids, mapa_classes)

        with open(caminho_txt, "w", encoding="utf-8") as f:
            for classe, cx, cy, w_, h_ in caixas:
                f.write(f"{classe} {cx:.6f} {cy:.6f} {w_:.6f} {h_:.6f}\n")

        registro.update({"indice": i, "arquivo": nome + ".png", "n_anotacoes": len(caixas)})
        escritor.writerow(registro)
        arq_meta.flush()
        gerados += 1

        if gerados % 25 == 0 or i == args.n - 1:
            dt = time.time() - inicio
            print(f"[{i + 1}/{args.n}] {gerados} geradas em {dt:.0f}s "
                  f"({dt / max(gerados, 1):.1f}s/img)")

    arq_meta.close()
    if os.path.exists(exr_tmp):
        os.remove(exr_tmp)
    print(f"CONCLUIDO: {gerados} novas imagens em {dir_pool}")


if __name__ == "__main__":
    main()
