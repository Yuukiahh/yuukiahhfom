"""Blender script: builds the 3D mascot "Nova" and renders its animation loops.

Run (CPU/EEVEE works headless; on the RTX 5070 Ti it is much faster):
    blender -b --factory-startup --python factory/mascot/build_mascot.py -- <out_dir> [--size 512] [--only idle_closed,wave]

Output: <out_dir>/<action>/0001.png ... (RGBA, transparent background, 60 fps).
factory/mascot/pack.py converts them into compact WebM (VP9+alpha) assets.

Design: floating egg-shaped robot, glossy white shell, black glass visor with emissive
cyan eyes/brows/mouth (expressions = swapped emissive shapes + brow pose), violet antenna
light, two floating hands, cyan belly ring and thruster glow. All geometry is procedural.
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Euler, Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
OUT = os.path.abspath(argv[0] if argv else "mascot_frames")
SIZE = int(argv[argv.index("--size") + 1]) if "--size" in argv else 512
ONLY = set(argv[argv.index("--only") + 1].split(",")) if "--only" in argv else None
ENGINE = argv[argv.index("--engine") + 1] if "--engine" in argv else "BLENDER_EEVEE"
FPS = 60

# ------------------------------------------------------------------ scene
bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = ENGINE
sc.render.resolution_x = sc.render.resolution_y = SIZE
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"
sc.render.image_settings.color_mode = "RGBA"
sc.render.fps = FPS
try:
    sc.view_settings.view_transform = "AgX"
    sc.view_settings.look = "AgX - Medium High Contrast"
except TypeError:
    sc.view_settings.view_transform = "Filmic"
if ENGINE == "BLENDER_EEVEE":
    e = sc.eevee
    e.taa_render_samples = 24
    e.use_bloom = True
    e.bloom_threshold = 0.9
    e.bloom_intensity = 0.06
    e.bloom_radius = 5.0
    e.use_gtao = True
    e.use_soft_shadows = True
else:
    sc.cycles.samples = 48
    sc.cycles.use_denoising = False

world = bpy.data.worlds.new("W")
sc.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.05, 0.06, 0.09, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6


def mat_plastic(name, color, rough=0.28, coat=0.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    for key in ("Coat Weight", "Clearcoat"):
        if key in b.inputs:
            b.inputs[key].default_value = coat
    return m


def mat_emit(name, color, strength):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs[0].default_value = (*color, 1)
    em.inputs[1].default_value = strength
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


M_SHELL = mat_plastic("shell", (0.88, 0.9, 0.95))
M_VISOR = mat_plastic("visor", (0.004, 0.005, 0.012), rough=0.06, coat=1.0)
M_EYE = mat_emit("eye", (0.1, 0.85, 1.0), 4.5)
M_RING = mat_emit("ring", (0.15, 0.75, 1.0), 6.0)
M_TIP = mat_emit("tip", (0.6, 0.15, 1.0), 5.0)
M_CHEEK = mat_emit("cheek", (1.0, 0.25, 0.55), 2.5)
M_THRUST = mat_emit("thrust", (0.3, 0.8, 1.0), 6.0)


def smooth(obj, levels=2):
    for p in obj.data.polygons:
        p.use_smooth = True
    if levels:
        mod = obj.modifiers.new("sub", "SUBSURF")
        mod.levels = mod.render_levels = levels


def sphere(name, loc, scale, mat, parent=None, seg=48):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=seg, ring_count=seg // 2, radius=1, location=loc)
    o = bpy.context.object
    o.name = name
    o.scale = scale
    o.data.materials.append(mat)
    smooth(o, 0)
    if parent:
        o.parent = parent
    return o


def arc(name, loc, radius, thickness, mat, parent, upper=True, scale=(1, 1, 1)):
    """Half torus facing the camera (-Y): smile (upper=False) or happy-eye arc (upper=True)."""
    bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=thickness, major_segments=48, minor_segments=12,
                                     location=(0, 0, 0), rotation=(math.pi / 2, 0, 0))
    o = bpy.context.object
    bpy.ops.object.transform_apply(rotation=True)
    bm = bmesh.new()
    bm.from_mesh(o.data)
    kill = [v for v in bm.verts if (v.co.z < -thickness * 0.2 if upper else v.co.z > thickness * 0.2)]
    bmesh.ops.delete(bm, geom=kill, context="VERTS")
    bm.to_mesh(o.data)
    bm.free()
    o.location = loc
    o.scale = scale
    o.name = name
    o.data.materials.append(mat)
    smooth(o, 0)
    o.parent = parent
    return o


root = bpy.data.objects.new("root", None)
sc.collection.objects.link(root)
body = sphere("body", (0, 0, 0), (1.0, 0.95, 0.92), M_SHELL, root, 64)
smooth(body, 1)
visor = sphere("visor", (0, -0.62, 0.12), (0.74, 0.42, 0.54), M_VISOR, body, 64)
bpy.ops.mesh.primitive_torus_add(major_radius=1.0, minor_radius=0.025, location=(0, 0, -0.38))
ring = bpy.context.object
ring.scale = (0.93, 0.88, 1)
ring.data.materials.append(M_RING)
ring.parent = body
smooth(ring, 0)
# antenna
bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=0.4, location=(0, 0, 1.05))
ant = bpy.context.object
ant.data.materials.append(M_SHELL)
ant.parent = body
tip = sphere("tip", (0, 0, 1.3), (0.075, 0.075, 0.075), M_TIP, body)
# thruster glow
thr = sphere("thruster", (0, 0, -0.98), (0.22, 0.22, 0.08), M_THRUST, root)

VY = -1.02  # visor front surface depth for features
EYE_X, EYE_Z = 0.26, 0.2
eyes = {}
for side, sx in (("L", -1), ("R", 1)):
    p = (sx * EYE_X, VY - 0.02, EYE_Z)
    eyes[("open", side)] = sphere(f"eye_open_{side}", p, (0.085, 0.03, 0.13), M_EYE, body)
    eyes[("wow", side)] = sphere(f"eye_wow_{side}", p, (0.115, 0.03, 0.165), M_EYE, body)
    eyes[("closed", side)] = sphere(f"eye_closed_{side}", (p[0], p[1], p[2] - 0.04), (0.1, 0.03, 0.02), M_EYE, body)
    eyes[("happy", side)] = arc(f"eye_happy_{side}", (p[0], p[1], p[2] - 0.05), 0.09, 0.026, M_EYE, body, upper=True)
    eyes[("squint", side)] = sphere(f"eye_squint_{side}", p, (0.09, 0.03, 0.05), M_EYE, body)
brows = {}
for side, sx in (("L", -1), ("R", 1)):
    b = sphere(f"brow_{side}", (sx * EYE_X, VY + 0.01, EYE_Z + 0.22), (0.1, 0.025, 0.022), M_EYE, body)
    brows[side] = b
cheeks = [sphere(f"cheek_{s}", (sx * 0.47, VY + 0.08, -0.02), (0.07, 0.02, 0.04), M_CHEEK, body)
          for s, sx in (("L", -1), ("R", 1))]
MZ = -0.12
mouths = {
    "smile": arc("mouth_smile", (0, VY + 0.01, MZ + 0.045), 0.14, 0.028, M_EYE, body, upper=False),
    "flat": sphere("mouth_flat", (0, VY + 0.01, MZ), (0.1, 0.02, 0.022), M_EYE, body),
    "open_s": sphere("mouth_open_s", (0, VY + 0.01, MZ), (0.09, 0.02, 0.045), M_EYE, body),
    "open_m": sphere("mouth_open_m", (0, VY + 0.01, MZ - 0.01), (0.115, 0.02, 0.075), M_EYE, body),
    "open_l": sphere("mouth_open_l", (0, VY + 0.0, MZ - 0.02), (0.14, 0.02, 0.105), M_EYE, body),
}
bpy.ops.mesh.primitive_torus_add(major_radius=0.075, minor_radius=0.022, location=(0, VY + 0.01, MZ - 0.02),
                                 rotation=(math.pi / 2, 0, 0))
mouths["o"] = bpy.context.object
mouths["o"].data.materials.append(M_EYE)
mouths["o"].parent = body
hands = {}
for side, sx in (("L", -1), ("R", 1)):
    h = sphere(f"hand_{side}", (sx * 1.3, -0.15, -0.25), (0.17, 0.15, 0.19), M_SHELL, root)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.13, minor_radius=0.018, location=(0, 0, 0), rotation=(0, math.pi / 2, 0))
    cuff = bpy.context.object
    cuff.data.materials.append(M_RING)
    cuff.parent = h
    cuff.scale = (6, 6.6, 5.3)  # compensates parent scale
    cuff.location = (-sx * 0.55, 0, 0)
    hands[side] = h

# lights + camera
def light(name, kind, loc, power, color=(1, 1, 1), size=3.0):
    d = bpy.data.lights.new(name, kind)
    d.energy = power
    d.color = color
    if kind == "AREA":
        d.size = size
    o = bpy.data.objects.new(name, d)
    o.location = loc
    o.rotation_euler = (Vector((0, 0, 0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    sc.collection.objects.link(o)
    return o


light("key", "AREA", (-3.5, -4.5, 3.5), 900)
light("fill", "AREA", (4, -3.5, 0.5), 250, (0.8, 0.9, 1.0))
light("rim", "AREA", (1.5, 4, 3), 1100, (0.45, 0.8, 1.0), 2.0)
light("rim2", "AREA", (-2.5, 3.5, 1), 600, (0.75, 0.45, 1.0), 2.0)
cam_d = bpy.data.cameras.new("cam")
cam_d.lens = 50
cam = bpy.data.objects.new("cam", cam_d)
cam.location = (0, -5.7, 0.3)
cam.rotation_euler = Euler((math.radians(88), 0, 0))
sc.collection.objects.link(cam)
sc.camera = cam


# ------------------------------------------------------------------ animation helpers
ALL_FEATURES = list(eyes.values()) + list(mouths.values()) + cheeks


BASE_T = {}


def snapshot():
    for o in bpy.data.objects:
        BASE_T[o.name] = (o.location.copy(), o.rotation_euler.copy(), o.scale.copy())


def clear_anim():
    for o in bpy.data.objects:
        o.animation_data_clear()
        if o.name in BASE_T:
            o.location, o.rotation_euler, o.scale = (x.copy() for x in BASE_T[o.name])
    for m in bpy.data.materials:
        if m.node_tree and m.node_tree.animation_data:
            m.node_tree.animation_data_clear()


def show(objs_on: set, frame: int):
    for o in ALL_FEATURES:
        hidden = o not in objs_on
        o.hide_render = hidden
        o.hide_viewport = hidden
        o.keyframe_insert("hide_render", frame=frame)
        o.keyframe_insert("hide_viewport", frame=frame)


def face(frame, eye="open", mouth="smile", cheeks_on=False):
    on = {eyes[(eye, "L")], eyes[(eye, "R")], mouths[mouth]}
    if cheeks_on:
        on |= set(cheeks)
    show(on, frame)


def key(obj, frame, loc=None, rot=None, scale=None):
    if loc is not None:
        obj.location = loc
        obj.keyframe_insert("location", frame=frame)
    if rot is not None:
        obj.rotation_euler = [math.radians(r) for r in rot]
        obj.keyframe_insert("rotation_euler", frame=frame)
    if scale is not None:
        obj.scale = scale
        obj.keyframe_insert("scale", frame=frame)


BASE_H = {s: Vector(h.location) for s, h in hands.items()}
BROW_BASE = {s: Vector(b.location) for s, b in brows.items()}


def brow(frame, raise_=0.0, tilt=0.0, side_tilt=None):
    for s, b in brows.items():
        sx = -1 if s == "L" else 1
        t = side_tilt.get(s, tilt) if side_tilt else tilt
        key(b, frame, loc=BROW_BASE[s] + Vector((0, 0, raise_)), rot=(0, sx * t, 0))


def idle_body(f, n, amp=1.0, phase=0.0):
    u = 2 * math.pi * (f / n) + phase
    key(root, f, loc=(0, 0, 0.07 * math.sin(u) * amp), rot=(0, 3 * math.sin(u + 1.1) * amp, 2 * math.sin(u * 0.5) * amp))
    key(body, f, rot=(2 * math.sin(u + 0.4) * amp, 0, 0))
    for s, h in hands.items():
        sx = -1 if s == "L" else 1
        key(h, f, loc=BASE_H[s] + Vector((0, 0, 0.07 * math.sin(u + (0.9 if sx > 0 else 0.0)))))
    key(thr, f, scale=(0.22, 0.22, 0.08 + 0.025 * abs(math.sin(u * 4))))


def ease(x):
    x = min(1, max(0, x))
    return x * x * (3 - 2 * x)


def bounce(x):
    x = min(1, max(0, x))
    return 1 - math.cos(x * math.pi * 3.5) * math.exp(-x * 5)


# ------------------------------------------------------------------ actions
def act_idle(mouth):
    n = 120
    for f in range(1, n + 1):
        idle_body(f, n)
        brow(f, 0.0, 0)
        blink = 78 <= f <= 84
        face(f, "closed" if blink else "open", mouth)
    return n


def act_wave():
    n = 96
    for f in range(1, n + 1):
        u = f / n
        idle_body(f, n, 0.6)
        up = ease(u / 0.2) * (1 - ease((u - 0.82) / 0.18))
        wav = math.sin((u - 0.15) * math.pi * 6) * up
        key(hands["R"], f, loc=BASE_H["R"] + Vector((0.15 * up + 0.12 * wav, -0.15 * up, 0.95 * up)),
            rot=(0, 25 * wav, 0))
        key(root, f, rot=(0, -6 * up, 0))
        brow(f, 0.05 * up, -8 * up)
        face(f, "happy" if 0.1 < u < 0.9 else "open", "open_m" if 0.15 < u < 0.6 else "smile", cheeks_on=0.1 < u < 0.9)
    return n


def act_point():
    n = 96
    for f in range(1, n + 1):
        u = f / n
        idle_body(f, n, 0.5)
        a = ease(u / 0.25) * (1 - ease((u - 0.85) / 0.15))
        key(hands["L"], f, loc=BASE_H["L"] + Vector((-0.35 * a, -0.45 * a, 0.55 * a)), scale=(0.17 + 0.08 * a, 0.15, 0.19 - 0.04 * a))
        key(root, f, rot=(0, 5 * a, -18 * a))
        for s in ("L", "R"):
            for k in ("open", "wow", "closed", "happy", "squint"):
                o = eyes[(k, s)]
                key(o, f, loc=(Vector(o.location) * 0) + Vector(((-1 if s == "L" else 1) * EYE_X - 0.07 * a,
                                                                o.location[1], EYE_Z + (-0.04 if k == "closed" else -0.05 if k == "happy" else 0) + 0.02 * a)))
        brow(f, 0.04 * a, -6 * a)
        face(f, "open", "smile" if u < 0.3 or u > 0.75 else "open_s")
    return n


def act_wow():
    n = 96
    for f in range(1, n + 1):
        u = f / n
        hop = math.sin(min(1, u / 0.35) * math.pi) if u < 0.35 else 0
        settle = bounce((u - 0.3) / 0.7) if u >= 0.3 else 0
        key(root, f, loc=(0, 0.35 * hop, 0.35 * hop + 0.03 * math.sin(u * 12)), rot=(-8 * hop, 0, 0))
        key(body, f, rot=(0, 0, 0), scale=(1.0 + 0.06 * hop, 0.95, 0.92 - 0.05 * hop))
        up = ease(u / 0.15) * (1 - ease((u - 0.8) / 0.2))
        for s, h in hands.items():
            sx = -1 if s == "L" else 1
            key(h, f, loc=BASE_H[s] + Vector((0.15 * sx * up, -0.2 * up, 0.75 * up + 0.05 * math.sin(u * 30) * up)))
        key(thr, f, scale=(0.22, 0.22, 0.08 + 0.15 * hop))
        brow(f, 0.12 * up, 12 * up)
        face(f, "wow" if 0.05 < u < 0.92 else "open", "o" if 0.05 < u < 0.85 else "smile")
    return n


def act_happy():
    n = 96
    for f in range(1, n + 1):
        u = f / n
        b = abs(math.sin(u * math.pi * 3)) * (1 - u * 0.4)
        key(root, f, loc=(0, 0, 0.22 * b), rot=(0, 6 * math.sin(u * math.pi * 3), 0))
        key(body, f, scale=(1.0 + 0.04 * (1 - b), 0.95, 0.92 - 0.04 * (1 - b)))
        clap = (math.sin(u * math.pi * 6) + 1) / 2
        for s, h in hands.items():
            sx = -1 if s == "L" else 1
            key(h, f, loc=Vector((sx * (0.55 + 0.55 * clap), -0.75, 0.05 + 0.15 * b)))
        brow(f, 0.06, -10)
        face(f, "happy", "open_l" if b > 0.4 else "smile", cheeks_on=True)
    return n


def act_think():
    n = 120
    for f in range(1, n + 1):
        u = f / n
        a = ease(u / 0.2) * (1 - ease((u - 0.88) / 0.12))
        idle_body(f, n, 0.4)
        key(hands["R"], f, loc=BASE_H["R"] + Vector((-0.75 * a, -0.65 * a, 0.05 * a + 0.03 * math.sin(u * 20) * a)))
        key(root, f, rot=(0, -8 * a, 10 * a))
        brow(f, 0.03 * a, 0, side_tilt={"L": 0, "R": 14 * a})
        face(f, "squint" if 0.15 < u < 0.85 else "open", "flat")
    return n


def act_intro():
    n = 72
    for f in range(1, n + 1):
        u = f / n
        k = ease(u / 0.6)
        overs = bounce((u - 0.45) / 0.55) if u > 0.45 else 0
        key(root, f, loc=(0, 0, -3.2 * (1 - k) + 0.08 * (1 - overs) * (u > 0.45)), rot=(0, 0, 360 * (1 - k)))
        key(thr, f, scale=(0.3, 0.3, 0.1 + 0.4 * (1 - k)))
        brow(f, 0.05, -6)
        face(f, "happy" if u > 0.65 else "open", "open_m" if u > 0.7 else "smile", cheeks_on=u > 0.65)
        for s, h in hands.items():
            key(h, f, loc=BASE_H[s] + Vector((0, 0, -0.3 * (1 - k))))
    return n


ACTIONS = {
    "idle_closed": lambda: act_idle("smile"),
    "idle_mid": lambda: act_idle("open_s"),
    "idle_open": lambda: act_idle("open_l"),
    "wave": act_wave,
    "point": act_point,
    "wow": act_wow,
    "happy": act_happy,
    "think": act_think,
    "intro": act_intro,
}

STILLS = "--stills" in argv
snapshot()
for name, fn in ACTIONS.items():
    if ONLY and name not in ONLY:
        continue
    clear_anim()
    n = fn()
    for fc in (ad.action.fcurves for o in bpy.data.objects if (ad := o.animation_data) and ad.action):
        for c in fc:
            for kp in c.keyframe_points:
                kp.interpolation = "CONSTANT" if c.data_path.startswith("hide") else "LINEAR"
    if STILLS:
        sc.frame_set(int(n * 0.5))
        sc.render.filepath = os.path.join(OUT, f"still_{name}.png")
        bpy.ops.render.render(write_still=True)
        print("MASCOT_DONE", name, "still", flush=True)
        continue
    sc.frame_start, sc.frame_end = 1, n
    os.makedirs(os.path.join(OUT, name), exist_ok=True)
    sc.render.filepath = os.path.join(OUT, name, "")
    bpy.ops.render.render(animation=True)
    print("MASCOT_DONE", name, n, flush=True)
