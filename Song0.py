# nebular_bloom_light_sheets.py
# Nebular Bloom: dark -> shimmering ambient evolution
# Requires: numpy
# Output: nebular_bloom_light_sheets.wav

import numpy as np, wave, os

# ---------- CONFIG ----------
sr = 44100
duration = 120            # seconds
num_sheets = 9
out_path = "nebular_bloom_light_sheets.wav"

# time vector
t = np.linspace(0, duration, int(sr*duration), endpoint=False)

# sheet fundamentals: deep base up to midrange for shimmer
base_freqs = np.linspace(40, 320, num_sheets)  # lower sheets and some higher for bloom
phase = np.random.RandomState(21).rand(num_sheets) * 2*np.pi
speed = 0.003 + np.random.RandomState(22).rand(num_sheets)*0.04
pan_positions = np.linspace(-1.0, 1.0, num_sheets)

def pan_gains(p):
    left = np.cos((p+1)*np.pi/4)
    right = np.sin((p+1)*np.pi/4)
    return left, right

def slow_lfo(rate, phase_offset):
    return 0.5*(1 + np.sin(2*np.pi*(rate*t) + phase_offset))

# output accumulators
out_l = np.zeros_like(t)
out_r = np.zeros_like(t)

# events controlling bloom intensity across time (a slow envelope that increases)
bloom_env = 0.2 + 0.8 * np.clip((t / duration)**1.2, 0, 1)   # slowly grows from 0.2 -> 1.0

# sparse "connect" events for structure (rare swells & shimmer bursts)
rng = np.random.RandomState(31)
event_times = np.sort(rng.rand(18) * duration)
event_width = 3.0   # large wide swells for nebula pulses

# cosmic high shimmer layer parameters (will be modulated by bloom_env)
def shimmer_layer(freq_base, width, phase0):
    # returns a gentle, bell-like shimmer at given center region
    tt = np.linspace(0, width, int(sr*width), endpoint=False)
    env = np.hanning(len(tt))**1.6
    # slightly inharmonic partials with fast micro-modulation
    freqs = freq_base * np.array([1.0, 1.997, 3.01, 4.05])
    sig = np.zeros_like(tt)
    for j, f in enumerate(freqs):
        sig += (0.5/(j+1)) * np.sin(2*np.pi*f*tt + phase0*(j+1) + 0.02*np.sin(2*np.pi*8*tt))
    return sig * env

# Build sheets
for i in range(num_sheets):
    bf = base_freqs[i]
    ph = phase[i]
    sp = speed[i]
    pan = pan_positions[i]
    gl, gr = pan_gains(pan)

    # LFOs controlling shape, brightness, shimmer tendency
    slow_shape = slow_lfo(sp*0.02, ph*0.6 + i*0.2)     # very slow shape
    bright = slow_lfo(sp*0.05, ph*1.1 + i*0.4)         # brightness for harmonics
    shimmer_ctrl = slow_lfo(sp*0.12, ph*0.3 + i*0.9)   # controls higher shimmer

    # slight frequency drift with slow beating (gives evolving interference)
    freq_dev = bf * (1 + 0.04*np.sin(2*np.pi*0.005*(i+1)*t + ph) +
                       0.02*np.sin(2*np.pi*0.013*(i+3)*t))

    # base pad voice: smooth low partials with lowpass-ish feel (by limiting harmonics)
    base = np.sin(2*np.pi*freq_dev*t + ph) * (0.9 - 0.6*shimmer_ctrl)
    h1 = 0.35 * np.sin(2*np.pi*(freq_dev*2.0)*t + ph*1.1) * (0.6*shimmer_ctrl + 0.4)
    h2 = 0.18 * np.sin(2*np.pi*(freq_dev*3.0)*t + ph*1.9) * (0.4*shimmer_ctrl)
    pad_sig = (base + h1 + h2)

    # gentle chorus-like micro-modulation for width & shimmer
    microfm = 0.005 * np.sin(2*np.pi*(0.5 + 0.1*i)*t + ph*0.7)
    pad_sig *= (1 + microfm)

    # connections: long resonant sweeps and shimmer bursts that are stronger as bloom grows
    connect_sig = np.zeros_like(t)
    for et in event_times:
        idx = np.searchsorted(event_times, et)
        mapped_pan = np.interp(idx, [0, len(event_times)-1], [-1, 1])
        dist = abs(pan - mapped_pan)
        strength = max(0, 1.0 - dist*1.6)
        start_idx = int(max(0, (et - event_width/2)*sr))
        end_idx = int(min(len(t), (et + event_width/2)*sr))
        if end_idx > start_idx:
            length = end_idx - start_idx
            win = np.hanning(length)
            # gentle rising sweep that widens with bloom
            sweep_base = bf * (0.6 + 0.8 * bloom_env[start_idx:end_idx])
            sweep = np.sin(2*np.pi * (sweep_base * np.linspace(1.0, 1.8, length)) * t[start_idx:end_idx])
            # add a shimmer burst overlay: short shimmering bells sprinkled over the event
            if strength > 0.05 and rng.rand() < 0.6:
                bell = shimmer_layer(bf*1.8, min(6.0, event_width*0.5), ph + i)
                # inject bell at start of event (windowed)
                b_len = len(bell)
                inject_len = min(b_len, length)
                connect_sig[start_idx:start_idx+inject_len] += bell[:inject_len] * (0.06 * strength * bloom_env[start_idx])
            connect_sig[start_idx:end_idx] += strength * sweep * win * (0.25 * bloom_env[start_idx:end_idx])

    # amplitude envelope: base slow envelope with bloom-driven lift towards the end
    env = 0.14 * (0.5 + 0.5*slow_lfo(0.012*(i+1), ph))    # base
    lift = 1.0 + 0.9 * bloom_env  # increases overall energy
    pad_sig = pad_sig * env * (0.8 + 0.35 * shimmer_ctrl) * (0.6 + 0.4*bloom_env)

    # shimmering upper layer: higher pitched wisps that grow with bloom_env
    shimmer_wisps = np.zeros_like(t)
    if i >= num_sheets//2:   # only generate wisps for upper sheets
        # random sparse triggers for wisps (density increases with bloom)
        n_trigs = int(6 + 12*bloom_env.mean() * (i/num_sheets))
        trig_times = np.linspace(0, duration, n_trigs+2)[1:-1] + 0.5*rng.randn(n_trigs)
        for tt0 in trig_times:
            if tt0 <= 0 or tt0 >= duration: continue
            dur_w = 1.5 + 4.0 * rng.rand() * bloom_env.mean()
            start = int(max(0, (tt0 - dur_w*0.5)*sr))
            end = int(min(len(t), (tt0 + dur_w*0.5)*sr))
            if end <= start: continue
            pos = np.linspace(0, dur_w, end-start, endpoint=False)
            # shimmering bell with FM
            bell = np.sin(2*np.pi*(bf*4.0 + 12.0*np.sin(2*np.pi*2.5*pos))*pos + ph)
            bell *= np.hanning(len(bell))**1.8
            shimmer_wisps[start:end] += 0.02 * bell * bloom_env[start:end]

    # pan & subtle interchannel delay to give width
    delay_samples = int((pan+1)*6)
    if delay_samples == 0:
        left = pad_sig * gl
        right = pad_sig * gr
    else:
        left = np.concatenate((np.zeros(delay_samples), pad_sig[:-delay_samples])) * gl
        right = np.concatenate((np.zeros(delay_samples), pad_sig[:-delay_samples])) * gr * 0.98

    out_l += left + connect_sig*gl + shimmer_wisps*gl
    out_r += right + connect_sig*gr + shimmer_wisps*gr

# subtle global cosmic noise, shaped so it becomes brighter with bloom
rng2 = np.random.RandomState(44)
noise = rng2.randn(len(t)) * 0.0009
# make noise a little warmer as bloom increases
noise *= (0.4 + 0.6 * bloom_env)
noise = np.convolve(noise, np.hanning(441), mode='same') * 0.6
out_l += noise
out_r += noise

# final gentle reverb/feedback delays (combination of short + medium + long)
def feedback_delay(sig, sr, delays_ms=[45, 120, 300], gains=[0.36, 0.28, 0.22]):
    out = sig.copy()
    for d_ms, g in zip(delays_ms, gains):
        d = int(sr * d_ms / 1000.0)
        if d < len(out):
            delayed = np.concatenate((np.zeros(d), out[:-d])) * g
            out += delayed * 0.4
    # smooth soft clip
    out = np.tanh(out * 1.05)
    return out

mix_l = feedback_delay(out_l, sr)
mix_r = feedback_delay(out_r, sr)

mix = np.vstack([mix_l, mix_r]).T
mx = np.max(np.abs(mix))
if mx > 0:
    mix = mix / (mx + 1e-9) * 0.95

# write WAV
with wave.open(out_path, 'wb') as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(sr)
    wf.writeframes((mix*32767).astype(np.int16).tobytes())

print(f"Rendered {duration}s of Nebular Bloom to: {os.path.abspath(out_path)}")
