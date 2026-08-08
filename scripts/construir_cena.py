"""
SinistroScan: construção procedural da cena no Blender.

Constrói um automóvel low-poly com as peças separadas por objeto (cada peça
carrega sua classe de detecção via propriedade customizada e um pass_index
único por instância), além de chão, céu físico, distratores e câmera.

A carroceria é um hatch compacto (proporções inspiradas no Renault Kwid:
~3,7 m de comprimento, capô curto e alto, traseira vertical com lanternas
em pé e rack de teto), para aproximar o domínio sintético das fotos reais
usadas na análise de domain gap.

Uso (headless):
    blender -b --factory-startup -P scripts/construir_cena.py -- [--render-teste]

Saídas:
    blender/cena.blend            cena configurada (entrega da etapa 3.2)
    docs/img/render_teste.png     render de conferência (com --render-teste)

Todos os campos de "stamp" do Blender são desligados: por padrão o Blender grava
nos metadados do PNG o caminho absoluto do .blend, a data e o nome da máquina —
o que vazaria dados do autor nas imagens publicadas do dataset.
"""

import math
import os
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

CLASSES = [
    "parachoque_dianteiro",
    "capo",
    "farol",
    "porta",
    "parachoque_traseiro",
    "lanterna",
    "retrovisor",
]


def _bsdf_de(mat):
    if mat.node_tree is None:
        try:
            mat.use_nodes = True
        except Exception:
            pass
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    saida = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputMaterial"), None)
    if saida is None:
        saida = nt.nodes.new("ShaderNodeOutputMaterial")
    if not saida.inputs["Surface"].links:
        nt.links.new(bsdf.outputs["BSDF"], saida.inputs["Surface"])
    return bsdf


def _set_input(no, nomes, valor):
    if isinstance(nomes, str):
        nomes = [nomes]
    for nome in nomes:
        if nome in no.inputs:
            no.inputs[nome].default_value = valor
            return True
    return False


def novo_material(nome, cor, rough=0.5, metal=0.0, coat=0.0):
    m = bpy.data.materials.new(nome)
    bsdf = _bsdf_de(m)
    _set_input(bsdf, "Base Color", (*cor, 1.0))
    _set_input(bsdf, "Roughness", rough)
    _set_input(bsdf, "Metallic", metal)
    _set_input(bsdf, ["Coat Weight", "Clearcoat"], coat)
    _set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.1)
    return m


def material_asfalto():
    m = bpy.data.materials.new("Asfalto")
    bsdf = _bsdf_de(m)
    nt = m.node_tree
    ruido = nt.nodes.new("ShaderNodeTexNoise")
    ruido.name = "AsfaltoRuido"
    _set_input(ruido, "Scale", 60.0)
    _set_input(ruido, "Detail", 6.0)
    rampa = nt.nodes.new("ShaderNodeValToRGB")
    rampa.name = "AsfaltoRampa"
    rampa.color_ramp.elements[0].color = (0.015, 0.015, 0.017, 1.0)
    rampa.color_ramp.elements[1].color = (0.09, 0.09, 0.095, 1.0)
    nt.links.new(ruido.outputs["Fac"], rampa.inputs["Fac"])
    nt.links.new(rampa.outputs["Color"], bsdf.inputs["Base Color"])
    _set_input(bsdf, "Roughness", 0.85)
    return m


def _finalizar(nome, bm, mat, colecao, bevel):
    mesh = bpy.data.meshes.new(nome)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(nome, mesh)
    colecao.objects.link(obj)
    if mat is not None:
        obj.data.materials.append(mat)
    if bevel > 0:
        mod = obj.modifiers.new("Bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = "ANGLE"
    return obj


def caixa(nome, minv, maxv, mat, colecao, bevel=0.02, taper_topo=None):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((
            minv[0] + (v.co.x + 0.5) * (maxv[0] - minv[0]),
            minv[1] + (v.co.y + 0.5) * (maxv[1] - minv[1]),
            minv[2] + (v.co.z + 0.5) * (maxv[2] - minv[2]),
        ))
    if taper_topo is not None:
        f_frente, f_tras, fy = taper_topo
        zmeio = (minv[2] + maxv[2]) / 2.0
        cx = (minv[0] + maxv[0]) / 2.0
        for v in bm.verts:
            if v.co.z > zmeio:
                fator = f_frente if v.co.x > cx else f_tras
                v.co.x = cx + (v.co.x - cx) * fator
                v.co.y *= fy
    return _finalizar(nome, bm, mat, colecao, bevel)


def cilindro_roda(nome, centro, raio, largura, mat, colecao):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=24,
                          radius1=raio, radius2=raio, depth=largura)
    rot = Matrix.Rotation(math.radians(90.0), 3, "X")
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=rot)
    bmesh.ops.translate(bm, verts=bm.verts, vec=Vector(centro))
    return _finalizar(nome, bm, mat, colecao, 0.0)


def construir():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    col_carro = bpy.data.collections.new("Carro")
    col_amb = bpy.data.collections.new("Ambiente")
    sc.collection.children.link(col_carro)
    sc.collection.children.link(col_amb)

    pintura = novo_material("Pintura", (0.45, 0.05, 0.07), rough=0.32, metal=0.70, coat=1.0)
    plastico = novo_material("PlasticoPreto", (0.012, 0.012, 0.013), rough=0.55)
    vidro = novo_material("VidroEscuro", (0.015, 0.02, 0.025), rough=0.06, metal=0.55)
    farol_vidro = novo_material("FarolVidro", (0.80, 0.84, 0.90), rough=0.12, metal=0.85)
    lanterna_mat = novo_material("LanternaVermelha", (0.52, 0.015, 0.025), rough=0.10, coat=1.0)
    cromado = novo_material("Cromado", (0.90, 0.90, 0.90), rough=0.15, metal=1.0)
    pneu = novo_material("PneuBorracha", (0.015, 0.015, 0.015), rough=0.95)
    aro = novo_material("RodaAro", (0.55, 0.56, 0.58), rough=0.35, metal=0.85)
    branco = novo_material("PlacaBranca", (0.88, 0.88, 0.88), rough=0.45)
    concreto = novo_material("Concreto", (0.42, 0.41, 0.40), rough=0.9)
    asfalto = material_asfalto()

    instancias = []

    caixa("corpo", (-1.70, -0.79, 0.30), (1.70, 0.79, 0.86), pintura, col_carro, bevel=0.05)
    caixa("cabine_vidro", (-1.66, -0.72, 0.84), (0.78, 0.72, 1.42), vidro, col_carro,
          bevel=0.03, taper_topo=(0.62, 0.96, 0.82))
    caixa("teto", (-1.56, -0.56, 1.395), (0.26, 0.56, 1.46), pintura, col_carro, bevel=0.03)
    caixa("tampa_traseira", (-1.72, -0.62, 0.55), (-1.64, 0.62, 1.30), pintura,
          col_carro, bevel=0.02)
    for lado_r, sr in (("dir", -1.0), ("esq", 1.0)):
        y0, y1 = sorted((sr * 0.44, sr * 0.52))
        caixa(f"rack_teto_{lado_r}", (-1.45, y0, 1.455), (0.20, y1, 1.50),
              plastico, col_carro, bevel=0.01)
    caixa("grade", (1.70, -0.32, 0.62), (1.74, 0.32, 0.80), plastico, col_carro, bevel=0.01)
    caixa("placa_dianteira", (1.84, -0.18, 0.48), (1.862, 0.18, 0.60), branco, col_carro, bevel=0.0)
    caixa("placa_traseira", (-1.862, -0.18, 0.48), (-1.84, 0.18, 0.60), branco, col_carro, bevel=0.0)

    for lado, s in (("dir", -1.0), ("esq", 1.0)):
        for pos, px in (("diant", 1.22), ("tras", -1.22)):
            cilindro_roda(f"pneu_{pos}_{lado}", (px, s * 0.70, 0.28), 0.28, 0.22, pneu, col_carro)
            cilindro_roda(f"aro_{pos}_{lado}", (px, s * 0.8175, 0.28), 0.14, 0.02, aro, col_carro)

    pd_a = caixa("parachoque_diant_pintado", (1.70, -0.83, 0.46), (1.84, 0.83, 0.68),
                 pintura, col_carro, bevel=0.03)
    pd_b = caixa("parachoque_diant_inferior", (1.70, -0.83, 0.24), (1.84, 0.83, 0.46),
                 plastico, col_carro, bevel=0.03)
    instancias.append(("parachoque_dianteiro", "parachoque_dianteiro", [pd_a, pd_b]))

    pt_a = caixa("parachoque_tras_pintado", (-1.84, -0.83, 0.46), (-1.70, 0.83, 0.68),
                 pintura, col_carro, bevel=0.03)
    pt_b = caixa("parachoque_tras_inferior", (-1.84, -0.83, 0.24), (-1.70, 0.83, 0.46),
                 plastico, col_carro, bevel=0.03)
    instancias.append(("parachoque_traseiro", "parachoque_traseiro", [pt_a, pt_b]))

    capo = caixa("capo", (0.78, -0.73, 0.86), (1.66, 0.73, 0.93), pintura, col_carro, bevel=0.02)
    instancias.append(("capo", "capo", [capo]))

    for lado, s in (("dir", -1.0), ("esq", 1.0)):
        y0, y1 = sorted((s * 0.34, s * 0.72))
        farol = caixa(f"farol_{lado}", (1.70, y0, 0.70), (1.82, y1, 0.84),
                      farol_vidro, col_carro, bevel=0.015)
        instancias.append((f"farol_{lado}", "farol", [farol]))

        y0, y1 = sorted((s * 0.58, s * 0.76))
        lant = caixa(f"lanterna_{lado}", (-1.82, y0, 0.66), (-1.70, y1, 1.06),
                     lanterna_mat, col_carro, bevel=0.015)
        instancias.append((f"lanterna_{lado}", "lanterna", [lant]))

        yin, yout = sorted((s * 0.79, s * 0.815))
        ymac0, ymac1 = sorted((s * 0.815, s * 0.835))
        pf = caixa(f"porta_diant_{lado}", (0.02, yin, 0.34), (0.88, yout, 0.84),
                   pintura, col_carro, bevel=0.015)
        mf = caixa(f"macaneta_diant_{lado}", (0.62, ymac0, 0.68), (0.78, ymac1, 0.715),
                   cromado, col_carro, bevel=0.0)
        instancias.append((f"porta_diant_{lado}", "porta", [pf, mf]))

        pt = caixa(f"porta_tras_{lado}", (-0.88, yin, 0.34), (-0.02, yout, 0.84),
                   pintura, col_carro, bevel=0.015)
        mt = caixa(f"macaneta_tras_{lado}", (-0.28, ymac0, 0.68), (-0.12, ymac1, 0.715),
                   cromado, col_carro, bevel=0.0)
        instancias.append((f"porta_tras_{lado}", "porta", [pt, mt]))

        yr0, yr1 = sorted((s * 0.74, s * 0.92))
        yh0, yh1 = sorted((s * 0.64, s * 0.74))
        concha = caixa(f"retrovisor_{lado}", (0.55, yr0, 0.99), (0.72, yr1, 1.10),
                       plastico, col_carro, bevel=0.012)
        haste = caixa(f"retrovisor_haste_{lado}", (0.59, yh0, 1.01), (0.66, yh1, 1.045),
                      plastico, col_carro, bevel=0.0)
        instancias.append((f"retrovisor_{lado}", "retrovisor", [concha, haste]))

    for i, (nome, classe, objs) in enumerate(instancias, start=1):
        for o in objs:
            o.pass_index = i
            o["classe"] = classe
            o["instancia"] = nome

    caixa("chao", (-40.0, -40.0, -0.05), (40.0, 40.0, 0.0), asfalto, col_amb, bevel=0.0)
    posicoes = [(9, 7), (-10, 8), (11, -6), (-8, -9), (0, 12)]
    for i, (dx, dy) in enumerate(posicoes, start=1):
        d = caixa(f"distrator_{i}", (dx - 1.5, dy - 1.5, 0.0), (dx + 1.5, dy + 1.5, 3.0),
                  concreto, col_amb, bevel=0.0)
        d.hide_render = True

    mundo = bpy.data.worlds.new("Mundo")
    sc.world = mundo
    if mundo.node_tree is None:
        try:
            mundo.use_nodes = True
        except Exception:
            pass
    nt = mundo.node_tree
    fundo = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeBackground"), None)
    if fundo is None:
        fundo = nt.nodes.new("ShaderNodeBackground")
    saida = next((n for n in nt.nodes if n.bl_idname == "ShaderNodeOutputWorld"), None)
    if saida is None:
        saida = nt.nodes.new("ShaderNodeOutputWorld")
    if not saida.inputs["Surface"].links:
        nt.links.new(fundo.outputs["Background"], saida.inputs["Surface"])
    ceu = nt.nodes.new("ShaderNodeTexSky")
    ceu.name = "Ceu"
    for tipo in ("MULTIPLE_SCATTERING", "SINGLE_SCATTERING", "NISHITA"):
        try:
            ceu.sky_type = tipo
            break
        except Exception:
            continue
    try:
        ceu.sun_disc = True
    except Exception:
        pass
    ceu.sun_elevation = math.radians(35.0)
    ceu.sun_rotation = math.radians(-35.0)
    nt.links.new(ceu.outputs["Color"], fundo.inputs["Color"])
    fundo.inputs["Strength"].default_value = 1.0

    alvo = bpy.data.objects.new("AlvoCamera", None)
    alvo.location = (0.0, 0.0, 0.55)
    sc.collection.objects.link(alvo)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 35.0
    cam = bpy.data.objects.new("Camera", cam_data)
    cam.location = (4.4, -3.8, 1.5)
    sc.collection.objects.link(cam)
    tr = cam.constraints.new("TRACK_TO")
    tr.target = alvo
    tr.track_axis = "TRACK_NEGATIVE_Z"
    tr.up_axis = "UP_Y"
    sc.camera = cam

    configurar_render(sc)

    sc["classes_json"] = __import__("json").dumps(CLASSES)
    return sc


def configurar_render(sc):
    sc.render.engine = "CYCLES"
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
        print("GPU:", prefs.compute_device_type,
              [d.name for d in prefs.devices if d.use])
    except Exception as e:
        print("AVISO: GPU indisponivel, usando CPU:", repr(e))
    sc.cycles.samples = 64
    sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.cycles.caustics_reflective = False
    sc.cycles.caustics_refractive = False
    sc.render.resolution_x = 640
    sc.render.resolution_y = 640
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGB"
    for prop in dir(sc.render):
        if prop.startswith("use_stamp"):
            try:
                setattr(sc.render, prop, False)
            except Exception:
                pass
    for vt in ("AgX", "Filmic", "Standard"):
        try:
            sc.view_settings.view_transform = vt
            break
        except Exception:
            continue
    sc.view_settings.exposure = -5.2
    print("VIEW_TRANSFORM:", sc.view_settings.view_transform,
          "EXPOSURE:", sc.view_settings.exposure)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    sc = construir()

    pasta_blend = os.path.join(RAIZ, "blender")
    os.makedirs(pasta_blend, exist_ok=True)
    caminho = os.path.join(pasta_blend, "cena.blend")
    bpy.ops.wm.save_as_mainfile(filepath=caminho)
    print("SALVO:", caminho)

    if "--render-teste" in argv:
        pasta_img = os.path.join(RAIZ, "docs", "img")
        os.makedirs(pasta_img, exist_ok=True)
        sc.render.filepath = os.path.join(pasta_img, "render_teste.png")
        bpy.ops.render.render(write_still=True)
        print("RENDER TESTE:", sc.render.filepath)


if __name__ == "__main__":
    main()
