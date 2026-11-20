"""CropGuard-Q console: a tkinter front-end for the quantum crop-integrity checker.

Run from the repository root:  .venv/bin/python cropguard_app.py
Everything runs locally with exact simulation. Optional extras, read from the environment only:
  CROPGUARD_LLM_URL / _KEY / _MODEL  let a language model write the analyst notes (challenge/analyst.py)
  QISKIT_IBM_TOKEN                   enables "submit new IBM job" (asks for confirmation first)
Saved studies (outputs/*.json) come from notebooks/CropGuard_Q.ipynb; the console only displays them.
"""
import json
import math
import os
import queue
import threading
import time
import tkinter as tk
from datetime import datetime, timezone
from tkinter import font as tkfont, messagebox

import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from challenge import analyst, guard
from challenge.data import ROOT, load_data
from challenge.kernel import QuantumKernelSVM, fidelity
from challenge.metrics import score
from challenge.submission import LABELS, MatchedControl, Submission

BG, PANEL, PANEL2, GRID = '#04070d', '#0a1220', '#0f1b2f', '#18304f'
CYAN, MAGENTA, GREEN, AMBER, RED, TEXT, DIM = '#00e5ff', '#ff2bd6', '#39ff88', '#ffb000', '#ff3860', '#cfe8ff', '#5f7a99'
CROP_COLORS = {'alfalfa': GREEN, 'maize': AMBER, 'wheat': CYAN}
OUT = ROOT / 'outputs'
matplotlib.rcParams.update({'figure.facecolor': PANEL, 'axes.facecolor': BG, 'axes.edgecolor': GRID,
                            'axes.labelcolor': TEXT, 'text.color': TEXT, 'xtick.color': DIM, 'ytick.color': DIM,
                            'grid.color': GRID, 'axes.grid': True, 'grid.alpha': .6, 'font.size': 8,
                            'legend.facecolor': PANEL, 'legend.edgecolor': GRID, 'axes.titlecolor': CYAN})


def read_json(name):
    try: return json.loads((OUT / name).read_text())
    except (OSError, ValueError): return None


class Console(tk.Tk):
    PAGES = [('mission', '◉  MISSION'), ('scanner', '⌖  FIELD SCANNER'), ('core', '⚛  QUANTUM CORE'),
             ('attack', '⚔  ATTACK LAB'), ('noise', '≋  NOISE + AI FIXER'), ('qpu', '▣  IBM QPU LINK')]

    def __init__(self):
        super().__init__()
        self.title('CropGuard-Q // Quantum Crop-Integrity Console')
        self.geometry('1440x900'); self.minsize(900, 640); self.configure(bg=BG)
        try: self.attributes('-zoomed', True)
        except tk.TclError: pass
        families = set(tkfont.families())
        mono = next((f for f in ['JetBrains Mono', 'Fira Code', 'Noto Sans Mono', 'DejaVu Sans Mono', 'Consolas',
                                 'Menlo', 'Liberation Mono'] if f in families), 'TkFixedFont')
        self.f = {k: (mono, s, w) for k, (s, w) in dict(xs=(8, 'normal'), s=(9, 'normal'), m=(10, 'normal'),
                                                         b=(10, 'bold'), l=(13, 'bold'), xl=(20, 'bold'),
                                                         xxl=(30, 'bold')).items()}
        self.jobs, self.t0, self.ready = queue.Queue(), time.time(), False
        self.adv, self.cur, self.phases = None, 0, np.zeros(8)
        self.header(); self.sidebar(); self.statusbar()
        self.main = tk.Frame(self, bg=BG); self.main.pack(side='left', fill='both', expand=True, padx=(0, 14), pady=10)
        self.pages = {}
        self.boot = tk.Text(self.main, bg=BG, fg=GREEN, font=self.f['m'], bd=0, highlightthickness=0)
        self.boot.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.after(60, self.poll); self.after(40, self.animate); self.after(500, self.tick)
        self.say('BOOT  initializing quantum crop-integrity console ...', CYAN)
        self.bg(self.load, self.loaded)

    # ---------- infrastructure ----------
    def bg(self, fn, done):
        """Run fn in a worker thread; done(result) runs back on the Tk thread."""
        def work():
            try: self.jobs.put((done, fn(), None))
            except Exception as e: self.jobs.put((done, None, e))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        while not self.jobs.empty():
            done, res, err = self.jobs.get()
            if err: self.say(f'ERROR {type(err).__name__}: {err}', RED)
            else: done(res)
        self.after(60, self.poll)

    def say(self, msg, color=TEXT):
        stamp = datetime.now().strftime('%H:%M:%S')
        self.status.configure(text=f'{stamp}  > {msg}', fg=color)
        if not self.ready:
            self.boot.insert('end', f'[{stamp}] {msg}\n'); self.boot.see('end')

    def button(self, parent, text, cmd, color=CYAN):
        b = tk.Button(parent, text=text, command=cmd, bg=PANEL2, fg=color, activebackground=color, activeforeground=BG,
                      relief='flat', bd=0, highlightthickness=1, highlightbackground=color, font=self.f['b'],
                      cursor='hand2', padx=12, pady=6)
        b.bind('<Enter>', lambda e: b.configure(bg=color, fg=BG)); b.bind('<Leave>', lambda e: b.configure(bg=PANEL2, fg=color))
        return b

    def panel(self, parent, title, grid=None, **pack):
        outer = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
        if grid: outer.grid(row=grid[0], column=grid[1], sticky='nsew', padx=(0, 8) if grid[1] == 0 else 0, pady=5)
        else: outer.pack(**({'fill': 'both', 'expand': True, 'pady': 5} | pack))
        tk.Label(outer, text='▸ ' + title.upper(), bg=PANEL, fg=CYAN, font=self.f['b'], anchor='w').pack(fill='x', padx=10, pady=(6, 2))
        tk.Frame(outer, bg=GRID, height=1).pack(fill='x', padx=10)
        inner = tk.Frame(outer, bg=PANEL); inner.pack(fill='both', expand=True, padx=10, pady=8)
        return inner

    def label(self, parent, text='', color=TEXT, size='m', wrap=False, **kw):
        lab = tk.Label(parent, text=text, bg=parent['bg'], fg=color, font=self.f[size], justify='left', anchor='w', **kw)
        if wrap: parent.bind('<Configure>', lambda e: lab.configure(wraplength=max(e.width - 8, 100)), add='+')
        return lab

    def figure(self, parent, w, h):
        fig = Figure(figsize=(w, h), dpi=100)
        canvas = FigureCanvasTkAgg(fig, parent)
        canvas.get_tk_widget().configure(bg=PANEL, highlightthickness=0)
        canvas.get_tk_widget().pack(fill='both', expand=True)
        return fig, canvas

    # ---------- chrome ----------
    def header(self):
        top = tk.Frame(self, bg=BG); top.pack(fill='x', padx=14, pady=(10, 0))
        left = tk.Frame(top, bg=BG); left.pack(side='left')
        tk.Label(left, text='◈ CROPGUARD-Q', bg=BG, fg=CYAN, font=self.f['xl']).pack(anchor='w')
        tk.Label(left, text='QUANTUM CROP-INTEGRITY CONSOLE  ·  DOUKKALA, MOROCCO  ·  SENTINEL-2  ·  25 APR 2024',
                 bg=BG, fg=DIM, font=self.f['s']).pack(anchor='w')
        right = tk.Frame(top, bg=BG); right.pack(side='right')
        self.clock = tk.Label(right, bg=BG, fg=TEXT, font=self.f['l']); self.clock.pack(anchor='e')
        self.pills = tk.Frame(right, bg=BG); self.pills.pack(anchor='e')
        self.pill_core = self.pill('CORE: BOOTING', AMBER)
        self.pill('8 QUBITS · EXACT', MAGENTA)
        self.scan = tk.Canvas(self, bg=BG, height=4, highlightthickness=0); self.scan.pack(fill='x', padx=14, pady=(8, 6))

    def pill(self, text, color):
        p = tk.Label(self.pills, text=f' {text} ', bg=BG, fg=color, font=self.f['xs'], highlightthickness=1,
                     highlightbackground=color, padx=6, pady=2)
        p.pack(side='left', padx=(6, 0)); return p

    def sidebar(self):
        side = tk.Frame(self, bg=PANEL, width=196, highlightthickness=1, highlightbackground=GRID)
        side.pack(side='left', fill='y', padx=(14, 10), pady=10); side.pack_propagate(False)
        tk.Label(side, text='MODULES', bg=PANEL, fg=DIM, font=self.f['xs']).pack(anchor='w', padx=14, pady=(14, 6))
        self.nav = {}
        for key, text in self.PAGES:
            row = tk.Frame(side, bg=PANEL); row.pack(fill='x', pady=1)
            bar = tk.Frame(row, bg=PANEL, width=4); bar.pack(side='left', fill='y')
            lab = tk.Label(row, text=text, bg=PANEL, fg=DIM, font=self.f['b'], anchor='w', padx=12, pady=9, cursor='hand2')
            lab.pack(side='left', fill='x', expand=True)
            lab.bind('<Button-1>', lambda e, k=key: self.show(k))
            lab.bind('<Enter>', lambda e, l=lab: l.configure(fg=TEXT) if l['fg'] == DIM else None)
            lab.bind('<Leave>', lambda e, l=lab, k=key: l.configure(fg=DIM) if self.nav[k][1]['bg'] == PANEL else None)
            self.nav[key] = (lab, bar)
        foot = tk.Frame(side, bg=PANEL); foot.pack(side='bottom', fill='x', padx=14, pady=12)
        for line, c in [('QISKIT FALL FEST', CYAN), ('EMSI × QUANTUM AFRICA', DIM), ('quantum · cybersec · AI', MAGENTA)]:
            tk.Label(foot, text=line, bg=PANEL, fg=c, font=self.f['xs']).pack(anchor='w')

    def statusbar(self):
        bar = tk.Frame(self, bg=PANEL, highlightthickness=1, highlightbackground=GRID); bar.pack(side='bottom', fill='x')
        self.status = tk.Label(bar, bg=PANEL, fg=TEXT, font=self.f['s'], anchor='w'); self.status.pack(side='left', padx=10, pady=4)

    def show(self, key):
        if not self.ready: return
        for k, (lab, bar) in self.nav.items():
            on = k == key
            lab.configure(fg=CYAN if on else DIM, bg=PANEL2 if on else PANEL); bar.configure(bg=CYAN if on else PANEL)
        self.pages[key].tkraise()

    def tick(self):
        self.clock.configure(text=datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M:%S UTC'))
        self.after(500, self.tick)

    def animate(self):
        t = time.time() - self.t0
        w = max(self.scan.winfo_width(), 10); x = (t * 260) % (w + 300) - 150
        self.scan.delete('all'); self.scan.create_line(0, 2, w, 2, fill=GRID, width=1)
        for i, c in enumerate([CYAN, '#00a6b8', '#006a78']):
            self.scan.create_line(x - i * 40, 2, x + 120 - i * 40, 2, fill=c, width=3 - i)
        if self.ready and hasattr(self, 'qcanvas'): self.draw_core(t)
        self.after(40, self.animate)

    # ---------- data + models ----------
    def load(self):
        log = lambda m: self.jobs.put((lambda _: self.say(m, GREEN), None, None))
        tables, _, manifest, _ = load_data(); F = manifest['features']
        log('[ OK ] release checksums, schema and split IDs verified (data/manifest.json)')
        tr, va = tables['train'], tables['validation']
        d = dict(F=F, X=tr[F].to_numpy(), y=tr.target.to_numpy(), Xv=va[F].to_numpy(), yv=va.target.to_numpy(),
                 ids=va.sample_id.to_numpy())
        t = time.perf_counter(); d['sub'] = Submission().fit(d['X'], d['y'], 0)
        log(f'[ OK ] PCA-8 fitted on 914 training fields; 914 eight-qubit states simulated ({time.perf_counter() - t:.1f} s)')
        d['ctrl'] = MatchedControl().fit(d['X'], d['y'], 0)
        log('[ OK ] matched classical control armed (polynomial SVM replaces the quantum SVM)')
        d['ins'] = d['sub'].inspect(d['Xv']); d['ins_c'] = d['ctrl'].inspect(d['Xv'])
        d['f1'] = {n: score(d['yv'], p, list(LABELS))['macro_f1'] for n, p in
                   [('quantum committee', d['ins']['prediction']), ('matched control', d['ins_c']['prediction']),
                    ('quantum SVM', d['ins']['votes'][0]), ('RBF SVM', d['ins']['votes'][1])]}
        d['mu'], d['sd'] = d['X'].mean(0), d['X'].std(0)
        d['studies'] = {n: read_json(n) for n in ['attack_lab.json', 'noise_study.json', 'hardware_result.json', 'hardware_result_robust.json',
                                                   'hardware_robustness.json', 'tuning_log.json', 'main_results.json']}
        log('[ OK ] saved studies loaded: ' + ', '.join(k for k, v in d['studies'].items() if v) or 'none yet')
        return d

    def loaded(self, d):
        self.d = d
        self.say('[ OK ] quantum committee online. Select a module.', CYAN)
        self.after(900, self.start)

    def start(self):
        self.ready = True; self.boot.destroy()
        self.pill_core.configure(text=' CORE: ONLINE ', fg=GREEN, highlightbackground=GREEN)
        for key, _ in self.PAGES:
            frame = tk.Frame(self.main, bg=BG); frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.pages[key] = frame; getattr(self, 'page_' + key)(frame)
        self.select(0); self.show('mission')

    # ---------- pages ----------
    def page_mission(self, root):
        d = self.d
        q = self.panel(root, 'headline question', fill='x', expand=False)
        self.label(q, 'Does adding a quantum model to a classical model add anything, especially when the data is tampered '
                      'with or incomplete, compared with adding another classical model instead?', TEXT, 'b', wrap=True).pack(anchor='w', fill='x')
        tiles = tk.Frame(root, bg=BG); tiles.pack(fill='x', pady=4)
        hw = d['studies']['hardware_result.json']
        for title, value, sub, color in [
                ('QUANTUM COMMITTEE', f"{d['f1']['quantum committee']:.3f}", 'validation macro-F1 · seed 0', CYAN),
                ('MATCHED CONTROL', f"{d['f1']['matched control']:.3f}", 'poly SVM replaces quantum', MAGENTA),
                ('OFFICIAL REFERENCE', '0.880', 'RBF SVM · 13 features', DIM),
                ('ALARM RATE (CLEAN)', f"{d['ins']['flag'].mean():.1%}", 'fields sent to a human', AMBER),
                ('IBM QPU', hw['backend'] if hw else 'not run', (hw['job_id'] if hw else 'set QISKIT_IBM_TOKEN'), GREEN)]:
            t = tk.Frame(tiles, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
            t.grid(row=0, column=len(tiles.grid_slaves()), sticky='nsew', padx=(0, 6))
            tiles.columnconfigure(len(tiles.grid_slaves()) - 1, weight=1, uniform='tile')
            tk.Frame(t, bg=color, height=3).pack(fill='x')
            self.label(t, title, DIM, 'xs').pack(anchor='w', padx=8, pady=(6, 0))
            self.label(t, value, color, 'xl').pack(anchor='w', padx=8)
            self.label(t, sub, DIM, 'xs', wrap=True).pack(anchor='w', fill='x', padx=8, pady=(0, 8))
        row = tk.Frame(root, bg=BG); row.pack(fill='both', expand=True)
        row.columnconfigure(0, weight=3); row.columnconfigure(1, weight=2); row.rowconfigure(0, weight=1)
        core = self.panel(row, 'quantum core · live phases of the selected field', grid=(0, 0))
        self.qcanvas = tk.Canvas(core, bg=PANEL, highlightthickness=0, height=200); self.qcanvas.pack(fill='both', expand=True)
        sysp = self.panel(row, 'system', grid=(0, 1))
        for line, c in [('13 Sentinel-2 bands (B1…B12, NDVI)', TEXT), ('   │  StandardScaler → PCA-8 → StandardScaler', DIM),
                        ('   │  (fitted on training fields only)', DIM), ('   ▼', DIM),
                        ('[ QUANTUM SVM ]  8-qubit ZZ fidelity kernel', CYAN), ('[ RBF SVM     ]  classical kernel', TEXT),
                        ('[ FOREST      ]  300 trees', TEXT), ('   │', DIM), ('   ▼  majority vote → crop label', GREEN),
                        ('   ▼  disagreement → TAMPER ALARM', RED), ('', TEXT),
                        ('AI error-fixer: linear regression maps noisy', MAGENTA),
                        ('QPU similarities back to exact ones.', MAGENTA), ('', TEXT),
                        ('Honest finding: the quantum partner is no more', AMBER),
                        ('robust than a classical one. See ATTACK LAB.', AMBER)]:
            self.label(sysp, line, c, 'xs').pack(anchor='w')

    def draw_core(self, t):
        c = self.qcanvas; c.delete('all')
        w, h = c.winfo_width(), c.winfo_height()
        if w < 50: return
        cx, cy = w / 2, h / 2 - 8; R = min(w, h) * .34; r = R * .3
        pos = [(cx + R * math.cos(2 * math.pi * i / 8 - math.pi / 2), cy + R * math.sin(2 * math.pi * i / 8 - math.pi / 2)) for i in range(8)]
        c.create_oval(cx - R * .45, cy - R * .45, cx + R * .45, cy + R * .45, outline=GRID, dash=(2, 4))
        c.create_text(cx, cy, text='|φ(x)⟩\n256 amplitudes', fill=DIM, font=self.f['xs'], justify='center')
        for i in range(7):  # linear entanglement: CX-phase-CX between neighbours q_i, q_i+1 (no q7-q0 link)
            (x1, y1), (x2, y2) = pos[i], pos[i + 1]
            glow = int(120 + 120 * (0.5 + 0.5 * math.sin(t * 3 + i)))
            c.create_line(x1, y1, x2, y2, fill=f'#{glow // 4:02x}{glow // 3:02x}{glow:02x}', width=2)
        for i, (x, y) in enumerate(pos):
            a = self.phases[i] + t * (0.6 + 0.08 * i)
            c.create_oval(x - r, y - r, x + r, y + r, fill=PANEL, outline=GRID, width=2)
            c.create_arc(x - r, y - r, x + r, y + r, start=math.degrees(a) % 360, extent=70, style='arc', outline=CYAN, width=3)
            c.create_line(x, y, x + r * math.cos(a), y - r * math.sin(a), fill=MAGENTA, width=2)
            c.create_oval(x - 3, y - 3, x + 3, y + 3, fill=CYAN, outline='')
            c.create_text(x, y + r + 11, text=f'q{i}  {self.phases[i]:+.2f}', fill=TEXT, font=self.f['xs'])
        c.create_text(10, h - 6, anchor='sw', fill=DIM, font=self.f['xs'],
                      text='needle = data phase 2·s·x_i + drift · links = linear ZZ entanglers')

    def page_scanner(self, root):
        d = self.d
        root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)
        col = tk.Frame(root, bg=BG); col.grid(row=0, column=0, sticky='nsw', padx=(0, 8))
        left = self.panel(col, 'validation fields')
        self.listbox = tk.Listbox(left, bg=BG, fg=TEXT, selectbackground=CYAN, selectforeground=BG, font=self.f['xs'],
                                  bd=0, highlightthickness=0, width=30, height=8, activestyle='none')
        self.listbox.pack(fill='both', expand=True)
        for i, sid in enumerate(d['ids']):
            mark = '⚠' if d['ins']['flag'][i] else ('✓' if d['ins']['prediction'][i] == d['yv'][i] else '✗')
            self.listbox.insert('end', f' {mark} {sid[:15]}  {d["yv"][i]}')
            self.listbox.itemconfig(i, fg=AMBER if mark == '⚠' else (TEXT if mark == '✓' else RED))
        self.listbox.bind('<<ListboxSelect>>', lambda e: self.select(self.listbox.curselection()[0]) if self.listbox.curselection() else None)
        self.label(left, '✓ verified  ⚠ flagged  ✗ wrong & unflagged', DIM, 'xs').pack(anchor='w', pady=(6, 0))
        ctl = self.panel(col, 'tamper injector', fill='x', expand=False)
        self.label(ctl, 'attack budget ε (L2, standardized bands)', DIM, 'xs').pack(anchor='w')
        self.eps = tk.DoubleVar(value=0.10)
        tk.Scale(ctl, from_=0.01, to=0.5, resolution=0.01, orient='horizontal', variable=self.eps, bg=PANEL,
                 fg=TEXT, troughcolor=GRID, highlightthickness=0, bd=0, activebackground=MAGENTA, sliderrelief='flat',
                 font=self.f['xs']).pack(fill='x')
        self.mode = tk.StringVar(value='naive')
        for v, t in [('naive', 'naive: fool the RBF SVM'), ('adaptive', 'adaptive: knows all 3 inspectors')]:
            tk.Radiobutton(ctl, text=t, value=v, variable=self.mode, bg=PANEL, fg=TEXT, selectcolor=BG, activebackground=PANEL,
                           activeforeground=CYAN, font=self.f['xs'], highlightthickness=0).pack(anchor='w')
        btns = tk.Frame(ctl, bg=PANEL); btns.pack(fill='x', pady=6)
        self.button(btns, 'INJECT TAMPER', self.inject, MAGENTA).pack(side='left', padx=(0, 6))
        self.button(btns, 'RESTORE', lambda: self.select(self.cur), GREEN).pack(side='left')
        self.button(ctl, 'AI ANALYST REPORT', self.analyse, CYAN).pack(fill='x')
        src = os.environ.get('CROPGUARD_LLM_URL')
        self.label(ctl, f"analyst: {'language model @ ' + src.split('/')[2] if src else 'offline template (set CROPGUARD_LLM_URL)'}",
                   DIM, 'xs', wrap=True).pack(anchor='w', fill='x', pady=(4, 0))
        right = tk.Frame(root, bg=BG); right.grid(row=0, column=1, sticky='nsew')
        self.banner = tk.Label(right, bg=PANEL, font=self.f['xl'], anchor='w', justify='left', padx=14, pady=6,
                               highlightthickness=1, highlightbackground=GRID)
        self.banner.pack(fill='x', pady=(5, 0))
        right.bind('<Configure>', lambda e: self.banner.configure(wraplength=max(e.width - 40, 200)), add='+')
        cards = tk.Frame(right, bg=BG); cards.pack(fill='x', pady=4)
        self.cards = []
        for k, name in enumerate(['QUANTUM SVM · 8 QUBITS', 'RBF SVM · CLASSICAL', 'RANDOM FOREST']):
            card = tk.Frame(cards, bg=PANEL, highlightthickness=1, highlightbackground=GRID)
            card.grid(row=0, column=k, sticky='nsew', padx=(0, 6 if k < 2 else 0)); cards.columnconfigure(k, weight=1, uniform='card')
            self.label(card, name, DIM, 'xs').pack(anchor='w', padx=8, pady=(6, 0))
            vote = self.label(card, '', TEXT, 'l'); vote.pack(anchor='w', padx=8)
            bars = tk.Canvas(card, bg=PANEL, height=58, highlightthickness=0); bars.pack(fill='x', padx=8, pady=(0, 8))
            self.cards.append((card, vote, bars))
        chart = self.panel(right, 'measurements · training-standardized units')
        self.band_fig, self.band_canvas = self.figure(chart, 6, 2.2)
        rep = self.panel(right, 'analyst incident note', fill='both', expand=False)
        self.report = tk.Text(rep, bg=BG, fg=GREEN, font=self.f['xs'], height=7, bd=0, highlightthickness=0, wrap='word')
        self.report.pack(fill='both', expand=True)

    def field_x(self):
        return self.adv if self.adv is not None else self.d['Xv'][self.cur]

    def select(self, i):
        self.cur, self.adv = i, None
        self.refresh()
        self.say(f'field {self.d["ids"][i]} loaded · surveyed crop: {self.d["yv"][i]}')

    def refresh(self):
        d, x = self.d, self.field_x()
        sub = d['sub']; Z = sub.transform(x[None])
        self.phases = 2 * sub.members[0].scale * Z[0]
        r = sub.inspect(x[None]); votes, final, flag = r['votes'][:, 0], r['prediction'][0], r['flag'][0]
        true = d['yv'][self.cur]
        color = RED if flag else (GREEN if final == true else AMBER)
        state = 'FLAGGED · inspectors disagree' if flag else 'VERIFIED · unanimous'
        self.banner.configure(text=f'{"⚠" if flag else "✓"}  {state}   →  {final.upper()}', fg=color, highlightbackground=color)
        scores = [sub.members[0].decision_function(Z)[0], sub.members[1].decision_function(Z)[0], sub.members[2].predict_proba(Z)[0]]
        for (card, vote, bars), v, s in zip(self.cards, votes, scores):
            ok = v == true
            card.configure(highlightbackground=CROP_COLORS[v]); vote.configure(text=v.upper() + ('' if ok else '  ✗'), fg=CROP_COLORS[v])
            share = s / s.sum() if s.min() >= 0 else np.exp(1.5 * s) / np.exp(1.5 * s).sum()  # display only
            bars.delete('all'); W = max(bars.winfo_width(), 200)
            for k, (lab, p) in enumerate(zip(LABELS, share)):
                y0 = 4 + k * 18
                bars.create_text(0, y0 + 6, text=lab[:7], anchor='w', fill=DIM, font=self.f['xs'])
                bars.create_rectangle(62, y0, W - 40, y0 + 12, outline=GRID)
                bars.create_rectangle(62, y0, 62 + (W - 102) * p, y0 + 12, fill=CROP_COLORS[lab], outline='')
                bars.create_text(W - 2, y0 + 6, text=f'{p:.2f}', anchor='e', fill=TEXT, font=self.f['xs'])
        fig = self.band_fig; fig.clear(); ax = fig.add_subplot(111)
        z0 = (d['Xv'][self.cur] - d['mu']) / d['sd']; z1 = (x - d['mu']) / d['sd']
        pos = np.arange(13)
        ax.bar(pos - .2, z0, .4, color=CYAN, label='recorded')
        ax.bar(pos + .2, z1, .4, color=MAGENTA if self.adv is not None else GRID, label='inspected')
        ax.set_xticks(pos, analyst.BANDS); ax.set_ylabel('SD from training mean'); ax.legend(loc='upper left', fontsize=7)
        fig.tight_layout(); self.band_canvas.draw_idle()
        self.info = dict(sample_id=str(d['ids'][self.cur]), true=str(true), final=str(final), flagged=bool(flag),
                         votes={'quantum_svm': str(votes[0]), 'rbf_svm': str(votes[1]), 'forest': str(votes[2])},
                         band_shift=(z1 - z0).tolist() if self.adv is not None else None, eps=self.eps.get())

    def inject(self):
        d, i, eps, mode = self.d, self.cur, round(self.eps.get(), 3), self.mode.get()
        x, yi = d['Xv'][i:i + 1], guard.index(d['yv'][i:i + 1])
        fn = guard.member_margin(d['sub'], 1) if mode == 'naive' else guard.committee_margin(d['sub'], d['Xv'], guard.index(d['yv']))
        self.say(f'injecting {mode} tamper, ε = {eps} ...', MAGENTA)
        self.bg(lambda: guard.tamper(fn, x, yi, eps, d['X'], iters=200, seed=0)[0], self.injected)

    def injected(self, adv):
        self.adv = adv; self.refresh()
        moved = np.linalg.norm((adv - self.d['Xv'][self.cur]) / self.d['sd'])
        self.say(f'tamper applied: {moved:.3f} SD moved · verdict {"FLAGGED" if self.info["flagged"] else "unanimous"} '
                 f'→ {self.info["final"]} (surveyed {self.info["true"]})', RED if self.info['flagged'] else AMBER)

    def analyse(self):
        info = dict(self.info)
        self.report.delete('1.0', 'end'); self.report.insert('end', 'analyst is writing ...')
        self.bg(lambda: analyst.note(info), self.analysed)

    def analysed(self, res):
        text, source = res
        self.report.delete('1.0', 'end'); self.report.insert('end', f'[{source}]\n{text}')
        self.say(f'analyst note ready ({source}); it explains, it never changes the verdict', CYAN)

    def page_core(self, root):
        d = self.d; q = d['sub'].members[0]
        top = tk.Frame(root, bg=BG); top.pack(fill='both', expand=True)
        top.columnconfigure(0, weight=3); top.columnconfigure(1, weight=2); top.rowconfigure(0, weight=1)
        heat = self.panel(top, 'kernel matrix · 150 validation fields sorted by crop', grid=(0, 0))
        self.k_fig, self.k_canvas = self.figure(heat, 4.6, 3.6)
        ctl = self.panel(top, 'rotation-strength knob', grid=(0, 1))
        self.label(ctl, 'Each 8-number field sets the angles of an 8-qubit ZZ feature map. The kernel is the overlap '
                        '|⟨φ(x)|φ(x′)⟩|² of two exact statevectors. Too strong a rotation makes every field look '
                        'unlike every other (kernel collapse).', TEXT, 'xs', wrap=True).pack(anchor='w', fill='x')
        self.scale = tk.DoubleVar(value=q.scale)
        tk.Scale(ctl, from_=0.02, to=1.0, resolution=0.005, orient='horizontal', variable=self.scale, bg=PANEL,
                 fg=TEXT, troughcolor=GRID, highlightthickness=0, bd=0, activebackground=MAGENTA, sliderrelief='flat',
                 font=self.f['xs'], label='scale s').pack(fill='x', pady=6)
        self.button(ctl, 'RECOMPUTE KERNEL', self.recompute, MAGENTA).pack(anchor='w')
        self.core_out = self.label(ctl, '', GREEN, 's', wrap=True); self.core_out.pack(anchor='w', fill='x', pady=8)
        r = q.resources()
        self.label(ctl, f"qubits {r['qubits']} · map depth {r['feature_map_depth']} · CX/map {r['feature_map_cx']}\n"
                        f"kernel circuit CX {r['kernel_circuit_cx']} · support vectors {r['support_vectors']}\n"
                        f"trainable quantum angles {r['trainable_quantum']} · exact statevector",
                   DIM, 'xs', wrap=True).pack(anchor='w', fill='x', side='bottom')
        tune = self.panel(root, 'tuning log · validation macro-F1 vs scale (30 configurations)', fill='both', expand=True)
        fig, canvas = self.figure(tune, 9, 1.9)
        log = d['studies']['tuning_log.json']
        ax = fig.add_subplot(111)
        if log:
            rows = log['quantum']['rows']
            for C in sorted({r['C'] for r in rows}):
                pts = sorted((r['scale'], r['validation_macro_f1']) for r in rows if r['C'] == C)
                ax.plot(*zip(*pts), marker='o', ms=3, label=f'C={C}')
            ax.legend(ncol=5, fontsize=7); ax.set_xlabel('scale s'); ax.set_ylabel('macro-F1')
        else:
            ax.text(.5, .5, 'run python -m challenge.tune to create outputs/tuning_log.json', ha='center', transform=ax.transAxes)
        fig.tight_layout(); canvas.draw_idle()
        self.order = np.argsort(d['yv'], kind='stable')
        self.recompute()

    def recompute(self):
        d, s = self.d, round(self.scale.get(), 3)
        C = d['sub'].members[0].svc.C
        self.core_out.configure(text='simulating 1064 statevectors ...', fg=AMBER)
        def work():
            Z, Zv = d['sub'].transform(d['X']), d['sub'].transform(d['Xv'])
            m = QuantumKernelSVM(scale=s, C=C).fit(Z, d['y'])
            S = m.states(Zv[self.order])
            return s, score(d['yv'], m.predict(Zv), list(LABELS))['macro_f1'], fidelity(S, S)
        self.bg(work, self.recomputed)

    def recomputed(self, res):
        s, f1, K = res
        off = K[~np.eye(len(K), dtype=bool)].mean()
        self.core_out.configure(text=f'scale {s}\nquantum SVM alone: macro-F1 {f1:.3f}\nmean off-diagonal kernel {off:.3f}',
                                fg=GREEN if f1 > .85 else RED)
        fig = self.k_fig; fig.clear(); ax = fig.add_subplot(111)
        im = ax.imshow(K, cmap='magma', vmin=0, vmax=1); ax.grid(False)
        ticks = [np.flatnonzero(self.d['yv'][self.order] == c).mean() for c in LABELS]
        ax.set_xticks(ticks, LABELS); ax.set_yticks(ticks, LABELS)
        fig.colorbar(im, ax=ax, fraction=.046); ax.set_title(f'fidelity kernel, s = {s}')
        fig.tight_layout(); self.k_canvas.draw_idle()
        self.say(f'kernel recomputed at s = {s}: macro-F1 {f1:.3f}', GREEN if f1 > .85 else RED)

    def page_attack(self, root):
        top = tk.Frame(root, bg=BG); top.pack(fill='x')
        self.button(top, 'RUN QUICK STUDY (seed 0, naive attacker)', self.quick_attack, MAGENTA).pack(side='left', pady=4)
        self.attack_note = self.label(top, '', DIM, 's'); self.attack_note.pack(side='left', padx=12)
        p = self.panel(root, 'tampered fields that slip through (all inspectors agree on a wrong crop)')
        self.a_fig, self.a_canvas = self.figure(p, 9, 3.4)
        self.adaptive_box = self.label(self.panel(root, 'adaptive attacker (knows every inspector)', fill='x', expand=False), '', TEXT, 's')
        self.adaptive_box.pack(anchor='w')
        lab = self.d['studies']['attack_lab.json']
        if lab: self.draw_attack(lab['tamper_rows'], 'saved study: outputs/attack_lab.json (notebook, seeds 0-2)')
        else: self.draw_attack([], 'no saved study yet: run the notebook or the quick study')

    def quick_attack(self):
        d = self.d
        self.say('running naive tamper study on 150 validation fields ...', MAGENTA)
        self.bg(lambda: guard.tamper_study(d['X'], d['y'], d['Xv'], d['yv'], seeds=(0,), adaptive=False)[0],
                lambda rows: self.draw_attack(rows, 'quick study: seed 0, naive attacker only (live)'))

    def draw_attack(self, rows, note):
        self.attack_note.configure(text=note)
        fig = self.a_fig; fig.clear(); ax = fig.add_subplot(111)
        colors = {'RBF SVM alone': DIM, '+ quantum SVM': CYAN, '+ poly SVM': MAGENTA, '+ forest': GREEN,
                  'quantum committee': '#7df9ff', 'matched control': '#ff8cf0'}
        for name, c in colors.items():
            pts = {}
            for r in rows:
                if r['alarm'] == name and r['attack'] in ('none', 'naive'): pts.setdefault(r['eps'], []).append(r['undetected'])
            if pts:
                e = sorted(pts); ax.plot(e, [np.mean(pts[k]) for k in e], marker='o', color=c, lw=2.4 if 'committee' in name or 'control' in name else 1.3,
                                         ls='-' if name != 'matched control' else '--', label=name)
        ax.set_xlabel('attack budget ε (standardized bands)'); ax.set_ylabel('share of fields: wrong & unflagged')
        ax.set_ylim(0, 1)
        if ax.lines: ax.legend(fontsize=7, ncol=3)
        fig.tight_layout(); self.a_canvas.draw_idle()
        ad = [r for r in rows if r['attack'] == 'adaptive']
        text = '\n'.join(f"ε={e:<5} " + '   '.join(f"{n}: {np.mean([r['undetected'] for r in ad if r['eps'] == e and r['alarm'] == n]):.0%} slip through"
                                                for n in ('quantum committee', 'matched control'))
                         for e in sorted({r['eps'] for r in ad}))
        self.adaptive_box.configure(text=text or 'adaptive results appear after the notebook run.')

    def page_noise(self, root):
        st = self.d['studies']['noise_study.json']
        top = self.panel(root, 'similarity error vs exact (mean absolute error, validation × training pairs)', fill='x', expand=False)
        if not st:
            self.label(top, 'No saved noise study. Install requirements-noise.txt and run the notebook.', AMBER).pack(anchor='w'); return
        self.label(top, '\n'.join(f"{r['level']}× noise   raw {r['mae_raw']:.3f}   ZNE {r['mae_zne']:.3f}   AI fixer {r['mae_fixed']:.3f}"
                                   for r in st['rows']), GREEN, 's').pack(anchor='w')
        self.label(top, f"subset: {st['n_train']} training × {st['n_val']} validation fields, {st['n_cal']} calibration pairs, "
                        'simulated device (qiskit-aer). The fixer is one straight line fitted on calibration pairs only.',
                   DIM, 'xs', wrap=True).pack(anchor='w', fill='x')
        row = tk.Frame(root, bg=BG); row.pack(fill='both', expand=True)
        row.columnconfigure((0, 1), weight=1, uniform='n'); row.rowconfigure(0, weight=1)
        p1 = self.panel(row, 'noisy (magenta) and AI-fixed (green) vs exact', grid=(0, 0))
        fig, canvas = self.figure(p1, 5, 4)
        r = st['rows'][-1]; ax = fig.add_subplot(111)
        ax.scatter(r['sample_exact'], r['sample_raw'], s=6, color=MAGENTA, label=f"raw, {r['level']}× noise")
        ax.scatter(r['sample_exact'], r['sample_fixed'], s=6, color=GREEN, label='after AI fixer')
        ax.plot([0, 1], [0, 1], color=DIM, lw=1); ax.set_xlabel('exact kernel value'); ax.set_ylabel('measured')
        ax.legend(fontsize=7); fig.tight_layout(); canvas.draw_idle()
        p2 = self.panel(row, 'error per method and noise level', grid=(0, 1))
        fig, canvas = self.figure(p2, 5, 4); ax = fig.add_subplot(111)
        lv = np.arange(len(st['rows']))
        for k, (key, c, name) in enumerate([('mae_raw', MAGENTA, 'raw'), ('mae_zne', AMBER, 'ZNE (2× circuits)'), ('mae_fixed', GREEN, 'AI fixer')]):
            ax.bar(lv + (k - 1) * .27, [r[key] for r in st['rows']], .27, color=c, label=name)
        ax.set_xticks(lv, [f"{r['level']}×" for r in st['rows']]); ax.set_ylabel('MAE'); ax.legend(fontsize=7)
        fig.tight_layout(); canvas.draw_idle()

    def page_qpu(self, root):
        st = self.d['studies']
        runs = [(n, st[f]) for n, f in [('submitted model · C=100', 'hardware_result.json'),
                                         ('hardware-ready model · C=1', 'hardware_result_robust.json')] if st.get(f)]
        top = self.panel(root, 'inference-only runs on IBM quantum hardware', fill='x', expand=False)
        token = bool(os.environ.get('QISKIT_IBM_TOKEN'))
        b = self.button(top, 'SUBMIT NEW INFERENCE JOB', self.submit_qpu, GREEN if token else DIM); b.pack(anchor='e')
        if not token: b.configure(state='disabled')
        if not runs:
            self.label(top, 'No saved hardware result (outputs/hardware_result*.json).', AMBER).pack(anchor='w'); return
        for name, hw in runs:
            agree = sum(f['quantum_hardware_fixed'] == f['quantum_exact'] for f in hw['fields'])
            self.label(top, f"{name}:  {hw['backend']} · job {hw['job_id']} · {hw['circuits']} circuits × {hw['shots']} shots · "
                            f"depth {hw['transpiled_depth']}, {hw['transpiled_two_qubit_gates']} two-qubit gates · kernel error "
                            f"raw {hw['mae_raw']:.3f} → AI fixer {hw['mae_fixed']:.3f} · quantum vote = exact on {agree}/{len(hw['fields'])} fields",
                       GREEN if agree == len(hw['fields']) else AMBER, 'xs', wrap=True).pack(anchor='w', fill='x', pady=1)
        row = tk.Frame(root, bg=BG); row.pack(fill='both', expand=True)
        row.columnconfigure((0, 1), weight=1, uniform='q'); row.rowconfigure(0, weight=1)
        p1 = self.panel(row, 'QPU vs exact simulator (after AI fixer: green)', grid=(0, 0))
        fig, canvas = self.figure(p1, 4.6, 3.4)
        for k, (name, hw) in enumerate(runs):
            ax = fig.add_subplot(1, len(runs), k + 1)
            n = hw['support_vectors'] * len(hw['fields'])
            ex, me = np.array(hw['exact']), np.array(hw['measured'])
            ax.scatter(ex[:n], me[:n], s=4, color=MAGENTA, label='raw')
            ax.scatter(ex[:n], np.clip(hw['fixer_slope'] * me[:n] + hw['fixer_intercept'], 0, 1), s=4, color=GREEN, label='fixed')
            ax.plot([0, 1], [0, 1], color=DIM, lw=1); ax.set_title(f"{hw['backend']} · C={name.split('C=')[1]}", fontsize=8)
            ax.set_xlabel('exact'); ax.legend(fontsize=6)
        fig.axes[0].set_ylabel('measured'); fig.tight_layout(); canvas.draw_idle()
        p2 = self.panel(row, 'why C matters: votes kept under kernel noise (offline scan)', grid=(0, 1))
        scan = st.get('hardware_robustness.json')
        if scan:
            fig, canvas = self.figure(p2, 4.6, 2.2); ax = fig.add_subplot(111)
            Cs = [r['C'] for r in scan['rows']]
            ax.semilogx(Cs, [r['agree_shot_noise'] for r in scan['rows']], marker='o', color=CYAN, label='512-shot noise')
            ax.semilogx(Cs, [r['agree_hardware_error'] for r in scan['rows']], marker='o', color=MAGENTA, label='measured QPU error')
            ax.semilogx(Cs, [r['exact_macro_f1'] for r in scan['rows']], marker='s', color=GREEN, ls='--', label='exact macro-F1')
            ax.set_xlabel('SVM C'); ax.set_ylim(0.3, 1.02); ax.legend(fontsize=6); fig.tight_layout(); canvas.draw_idle()
        for name, hw in runs:
            for f in hw['fields']:
                ok = f['quantum_hardware_fixed'] == f['quantum_exact']
                self.label(p2, f"C={name.split('C=')[1]} · {f['name']}: exact {f['quantum_exact']} · QPU+fixer {f['quantum_hardware_fixed']}"
                               f" · RBF {f['rbf']} · forest {f['forest']}", GREEN if ok else AMBER, 'xs', wrap=True).pack(anchor='w', fill='x')
        self.label(p2, runs[0][1]['scope'], DIM, 'xs', wrap=True).pack(anchor='w', fill='x', pady=(4, 0))

    def submit_qpu(self):
        if not messagebox.askyesno('IBM Quantum', 'Submit an inference-only job (4 fields × support vectors + 40 calibration '
                                                  'pairs, 512 shots) on the least busy IBM backend? It uses QPU time from your plan.'):
            return
        from challenge import hardware_kernel as hk
        d = self.d
        self.say('submitting inference-only job to IBM Quantum ... (queue time varies)', GREEN)
        def work():
            fields = hk.pick_fields(d['sub'], d['X'], d['Xv'], d['yv'], d['ids'])
            return hk.run(d['sub'], fields, hk.connect(), shots=512, n_cal=40, approved=True)
        self.bg(work, lambda hw: self.say(f"QPU job {hw['job_id']} done on {hw['backend']}: kernel MAE raw {hw['mae_raw']:.3f} → "
                                          f"fixed {hw['mae_fixed']:.3f}. Restart the console to view it.", GREEN))


if __name__ == '__main__':
    Console().mainloop()
