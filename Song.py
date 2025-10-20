from pyo import *
from vpython import *
import random
import math

# -------------------------------
# 1. Initialize audio
# -------------------------------
s = Server().boot()
s.start()

wave = Sine(freq=20, mul=0.2)
wave_mod = wave * Sine(freq=0.01, mul=0.05, add=1)
wave_pan = Pan(wave_mod, outs=2, pan=Sine(freq=0.005, mul=0.5, add=0.5))
wave_rev = Freeverb(wave_pan, size=0.9, damp=0.5, bal=0.3).out()

amp_follow = Follower(wave_mod, freq=50)
low_filter = Biquad(wave_mod, freq=100, q=1, type=0)
mid_filter = Biquad(wave_mod, freq=500, q=1, type=0)
high_filter = Biquad(wave_mod, freq=2000, q=1, type=0)
low_amp = Follower(low_filter, freq=50)
mid_amp = Follower(mid_filter, freq=50)
high_amp = Follower(high_filter, freq=50)

# -------------------------------
# 2. Initialize visuals
# -------------------------------
scene = canvas(title="Optimized Sonic Multiverse", width=900, height=700, background=color.black)

crystals = []
crystal_vel = []
sheets = []
sheet_offsets = []

particles = []
glows = []
nebula_trails = []

MAX_PARTICLES = 300
MAX_NEBULA = 100

distant_light(direction=vector(0,1,0), color=color.white*0.2)
distant_light(direction=vector(1,1,0), color=color.cyan*0.1)

# -------------------------------
# 3. Interactive brush state
# -------------------------------
mouse_pressed = False
def mouse_down(evt): global mouse_pressed; mouse_pressed = True
def mouse_up(evt): global mouse_pressed; mouse_pressed = False
scene.bind('mousedown', mouse_down)
scene.bind('mouseup', mouse_up)

# -------------------------------
# 4. Spawn particles (optimized)
# -------------------------------
def spawn_brush_particles(pos, amp_mult=1.0):
    if len(particles) >= MAX_PARTICLES:
        oldest = particles.pop(0)
        oldest.visible = False
    la, ma, ha = low_amp.get()*amp_mult, mid_amp.get()*amp_mult, high_amp.get()*amp_mult
    color_p = vector(min(1,la*5), min(1,ma*5), min(1,ha*5))
    scale_factor = 0.03 + 0.05*amp_follow.get()*amp_mult
    p = sphere(pos=pos, radius=scale_factor, color=color_p, opacity=0.8, emissive=True)
    p.vel = vector(random.uniform(-0.02,0.02), random.uniform(-0.02,0.02), random.uniform(-0.02,0.02))
    p.trail = []
    particles.append(p)
    # Glow layers
    for scale in [1.5,2.0]:
        g = sphere(pos=pos, radius=p.radius*scale, color=color_p, opacity=0.05+0.03*amp_follow.get()*amp_mult, emissive=True)
        g.vel = vector(random.uniform(-0.01,0.01), random.uniform(-0.01,0.01), random.uniform(-0.01,0.01))
        glows.append(g)

# -------------------------------
# 5. Fracture sheets with nebula trails (optimized)
# -------------------------------
def fracture_sheet(s):
    fragments = []
    for _ in range(3):
        frag_pos = s.pos + vector(random.uniform(-s.size.x/4,s.size.x/4),
                                  random.uniform(-0.02,0.02),
                                  random.uniform(-s.size.z/4,s.size.z/4))
        frag_size = s.size*0.5
        color_frag = vector(min(1,low_amp.get()*5), min(1,mid_amp.get()*5), min(1,high_amp.get()*5))
        frag = box(pos=frag_pos, size=frag_size, color=color_frag, opacity=s.opacity, emissive=True)
        sheet_offsets.append(random.uniform(0,2*math.pi))
        fragments.append(frag)
        # Particle explosion
        for _ in range(5):
            spawn_brush_particles(frag.pos, amp_mult=2.0)
        # Nebula trail
        if len(nebula_trails) >= MAX_NEBULA:
            oldest_n = nebula_trails.pop(0)
            oldest_n.visible = False
        nebula = sphere(pos=frag.pos, radius=frag_size.x, color=color_frag, opacity=0.1, emissive=True)
        nebula.vel = vector(random.uniform(-0.01,0.01), random.uniform(-0.01,0.01), random.uniform(-0.01,0.01))
        nebula_trails.append(nebula)
    s.visible = False
    return fragments

# -------------------------------
# 6. Animate multiverse (optimized)
# -------------------------------
t = 0
frame_skip = 0
while True:
    rate(60)
    frame_skip += 1
    current_amp = amp_follow.get()
    bass_amp = low_amp.get()
    
    # Randomly spawn crystals & sheets
    if len(crystals) < 20 and random.random() < 0.01:
        pos = vector(random.uniform(-5,5), random.uniform(-5,5), random.uniform(-5,5))
        c = pyramid(pos=pos, size=vector(0.1,0.1,0.1), color=vector(random.random(),random.random(),random.random()), emissive=True)
        crystals.append(c)
        crystal_vel.append(vector(random.uniform(-0.02,0.02), random.uniform(-0.02,0.02), random.uniform(-0.02,0.02)))
    if len(sheets) < 5 and random.random() < 0.005:
        pos = vector(random.uniform(-5,5), random.uniform(-5,5), random.uniform(-5,5))
        s = box(pos=pos, size=vector(1,0.05,1), color=vector(random.random(),random.random(),random.random()), opacity=0.4, emissive=True)
        sheets.append(s)
        sheet_offsets.append(random.uniform(0,2*math.pi))
    
    # Animate crystals every 2 frames
    if frame_skip % 2 == 0:
        for i, c in enumerate(crystals):
            c.rotate(angle=crystal_vel[i].mag, axis=crystal_vel[i])
            c.pos += vector(0.002*math.sin(t), 0.002*math.cos(t), 0)
            c.size = vector(0.1 + current_amp*3, 0.1 + current_amp*3, 0.1 + current_amp*3)
            c.color = vector(min(1,low_amp.get()*5), min(1,mid_amp.get()*5), min(1,high_amp.get()*5))
    
    # Animate sheets & fracture
    new_sheets = []
    for idx, s in enumerate(sheets):
        wobble = 0.1*math.sin(t+sheet_offsets[idx])*current_amp
        scale_factor = 1 + 0.5*current_amp*math.sin(t+s.pos.x+s.pos.y)
        s.size.x = scale_factor*2 + wobble
        s.size.z = scale_factor*2 + wobble
        s.opacity = 0.2 + 0.6*current_amp
        s.color = vector(min(1,low_amp.get()*5), min(1,mid_amp.get()*5), min(1,high_amp.get()*5))
        if bass_amp > 0.2 and random.random() < 0.05:
            new_sheets.extend(fracture_sheet(s))
        else:
            new_sheets.append(s)
    sheets = new_sheets
    
    # Continuous brush
    if mouse_pressed and scene.mouse.pos is not None:
        spawn_brush_particles(scene.mouse.project(normal=vector(0,0,1)))
    
    # Particle trails & physics
    for i, p1 in enumerate(particles):
        for j, p2 in enumerate(particles[i+1:], start=i+1):
            dir_vec = p1.pos - p2.pos
            dist = mag(dir_vec)+1e-5
            if dist < 0.1:
                force = 0.0005/dist
                p1.vel += norm(dir_vec)*force
                p2.vel -= norm(dir_vec)*force
        p1.pos += p1.vel
        if frame_skip % 2 == 0:
            trail_s = sphere(pos=p1.pos, radius=p1.radius*0.3, color=p1.color, opacity=0.2, emissive=True)
            p1.trail.append(trail_s)
            if len(p1.trail) > 5:
                old = p1.trail.pop(0)
                old.visible = False
        p1.opacity *= 0.96
        if p1.opacity < 0.05:
            p1.visible = False
            particles[i] = None
    particles = [p for p in particles if p is not None]
    
    # Glow layers
    for g in glows[:]:
        g.pos += g.vel
        g.opacity *= 0.95
        if g.opacity < 0.01:
            g.visible = False
            glows.remove(g)
    
    # Nebula trails
    for n in nebula_trails[:]:
        n.pos += n.vel
        n.opacity *= 0.97
        if n.opacity < 0.01:
            n.visible = False
            nebula_trails.remove(n)
    
    # Camera update every 2 frames
    if frame_skip % 2 == 0:
        cam_amp = min(0.5,current_amp)
        scene.camera.pos = vector(15*math.sin(t*0.02)*(1+cam_amp*3),
                                  10*math.sin(t*0.01)*(1+cam_amp*2),
                                  15*math.cos(t*0.02)*(1+cam_amp*3))
        scene.camera.axis = vector(-scene.camera.pos.x,-scene.camera.pos.y,-scene.camera.pos.z)
    
    t += 0.05
