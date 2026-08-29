from __future__ import annotations

import argparse
import math
import queue
import threading
import traceback
from pathlib import Path

import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from transformers import AutoImageProcessor, ViTMAEForPreTraining


class PulsingMAE:
    """Frozen ViT-MAE with inference-time residual gain hooks.

    For encoder block l, replace ordinary output B_l(h) with

        h + g_l(t) * (B_l(h) - h)

    g=1 is the pretrained execution, g=0 bypasses a block,
    and g>1 extrapolates along its learned update.
    """

    def __init__(self, model_name: str, device: str):
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = ViTMAEForPreTraining.from_pretrained(model_name).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Transformers changed ViT-MAE internals: current releases expose
        # encoder blocks as model.vit.layers; older releases used encoder.layer.
        if hasattr(self.model.vit, "layers"):
            self.layers = list(self.model.vit.layers)
            self.layer_layout = "vit.layers"
        elif hasattr(self.model.vit, "encoder") and hasattr(self.model.vit.encoder, "layer"):
            self.layers = list(self.model.vit.encoder.layer)
            self.layer_layout = "vit.encoder.layer"
        else:
            names = [name for name, _ in self.model.vit.named_children()]
            raise AttributeError(
                "Could not locate ViT-MAE encoder blocks. "
                f"vit children are: {names}"
            )

        self.gains = [1.0] * len(self.layers)
        self.handles = []
        self.last_signature = None
        self._install_hooks()

        patch_size = self.model.config.patch_size
        self.patch_size = patch_size if isinstance(patch_size, int) else int(patch_size[0])
        self.num_channels = int(self.model.config.num_channels)
        self.mean = torch.tensor(self.processor.image_mean, device=self.device).view(1, -1, 1, 1)
        self.std = torch.tensor(self.processor.image_std, device=self.device).view(1, -1, 1, 1)

    def _install_hooks(self):
        for idx, layer in enumerate(self.layers):
            def hook(module, inputs, output, idx=idx):
                h_in = inputs[0]
                if isinstance(output, tuple):
                    h_out = output[0]
                    scaled = h_in + float(self.gains[idx]) * (h_out - h_in)
                    if idx == len(self.layers) - 1:
                        self.last_signature = scaled.mean(dim=1).detach()
                    return (scaled,) + output[1:]

                h_out = output
                scaled = h_in + float(self.gains[idx]) * (h_out - h_in)
                if idx == len(self.layers) - 1:
                    self.last_signature = scaled.mean(dim=1).detach()
                return scaled

            self.handles.append(layer.register_forward_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()

    def set_gains(self, gains):
        if len(gains) != len(self.layers):
            raise ValueError("gain count must match encoder layer count")
        self.gains = [float(g) for g in gains]

    def _set_mask_ratio(self, ratio):
        ratio = float(np.clip(ratio, 0.0, 0.95))
        self.model.config.mask_ratio = ratio
        if hasattr(self.model.vit, "config"):
            self.model.vit.config.mask_ratio = ratio
        if hasattr(self.model.vit.embeddings, "config"):
            self.model.vit.embeddings.config.mask_ratio = ratio

    def _unpatchify(self, logits):
        b, n, dim = logits.shape
        p = self.patch_size
        c = self.num_channels
        if dim != p * p * c:
            raise RuntimeError(f"unexpected decoder dimension {dim}; expected {p*p*c}")

        gh = int(round(math.sqrt(n)))
        if gh * gh != n:
            raise RuntimeError(f"expected square patch grid, got {n} patches")

        x = logits.reshape(b, gh, gh, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(b, c, gh * p, gh * p)

    @torch.inference_mode()
    def reconstruct(self, image: Image.Image, mask_ratio: float, seed: int):
        self._set_mask_ratio(mask_ratio)
        batch = self.processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = batch["pixel_values"].to(self.device)

        ph = pixel_values.shape[-2] // self.patch_size
        pw = pixel_values.shape[-1] // self.patch_size
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        noise = torch.rand((pixel_values.shape[0], ph * pw), generator=gen).to(self.device)

        out = self.model(pixel_values=pixel_values, noise=noise, return_dict=True)
        recon = self._unpatchify(out.logits)

        # facebook/vit-mae-base uses norm_pix_loss=False. The decoder predicts
        # the processor-normalized pixel space, so return to display RGB here.
        recon = recon * self.std + self.mean
        recon = recon.clamp(0.0, 1.0)

        arr = recon[0].permute(1, 2, 0).detach().cpu().numpy()
        pil = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")

        if self.last_signature is None:
            signature = np.zeros(1, dtype=np.float32)
        else:
            signature = self.last_signature[0].float().cpu().numpy()

        return pil, signature


class DreamApp:
    def __init__(self, root, model_name: str, device: str):
        self.root = root
        self.root.title("Pulsing Transformer Dream")
        self.root.geometry("1320x900")
        self.root.configure(bg="#0a0e13")

        self.model_name = model_name
        self.device = device
        self.engine = None
        self.result_queue = queue.Queue()
        self.busy = False
        self.running = False
        self.step_index = 0
        self.seed_image = None
        self.current_image = None
        self.seed_signature = None
        self.current_signature = None

        self.base_var = tk.DoubleVar(value=1.0)
        self.amp_var = tk.DoubleVar(value=0.70)
        self.period_var = tk.DoubleVar(value=12.0)
        self.depth_var = tk.DoubleVar(value=1.0)
        self.blend_var = tk.DoubleVar(value=0.72)
        self.mask_var = tk.DoubleVar(value=0.20)
        self.noise_var = tk.DoubleVar(value=0.0)
        self.mode_var = tk.StringVar(value="wave")

        self._setup_ui()
        self._load_engine_async()
        self.root.after(80, self._poll)

    def _setup_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background="#0a0e13")
        style.configure("TLabel", background="#0a0e13", foreground="#dbe7ee")
        style.configure("TLabelframe", background="#0a0e13", foreground="#dbe7ee")
        style.configure("TLabelframe.Label", background="#0a0e13", foreground="#8ccad8")
        style.configure("TButton", font=("Segoe UI", 9, "bold"))

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(
            top,
            text="PULSING TRANSFORMER DREAM",
            font=("Segoe UI", 18, "bold"),
            foreground="#70f0d2",
        ).pack(side=tk.LEFT)
        self.status = ttk.Label(top, text="Loading frozen transformer…", foreground="#f0c96a")
        self.status.pack(side=tk.RIGHT)

        controls = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        controls.pack(fill=tk.X)
        ttk.Button(controls, text="LOAD IMAGE", command=self.load_image).pack(side=tk.LEFT, padx=3)
        self.start_btn = ttk.Button(controls, text="START", command=self.toggle_run, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=3)
        self.step_btn = ttk.Button(controls, text="STEP", command=self.step_once, state=tk.DISABLED)
        self.step_btn.pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="RESET IMAGE", command=self.reset_image).pack(side=tk.LEFT, padx=3)
        ttk.Button(controls, text="SAVE CURRENT", command=self.save_current).pack(side=tk.LEFT, padx=3)

        ttk.Separator(controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=9)
        for text, mode in [
            ("ALL PASS", "flat"),
            ("WAVE", "wave"),
            ("MIDDLE", "middle"),
            ("ALTERNATE", "alternate"),
            ("STROBE", "strobe"),
        ]:
            ttk.Button(controls, text=text, command=lambda m=mode: self.mode_var.set(m)).pack(side=tk.LEFT, padx=2)

        sliders = ttk.LabelFrame(self.root, text="Execution schedule", padding=8)
        sliders.pack(fill=tk.X, padx=10, pady=(0, 8))
        specs = [
            ("base", self.base_var, -0.5, 2.0),
            ("pulse amp", self.amp_var, 0.0, 1.8),
            ("period", self.period_var, 2.0, 40.0),
            ("depth cycles", self.depth_var, 0.0, 3.0),
            ("image blend", self.blend_var, 0.0, 1.0),
            ("mask ratio", self.mask_var, 0.0, 0.9),
            ("pixel noise", self.noise_var, 0.0, 0.12),
        ]

        for i, (name, var, lo, hi) in enumerate(specs):
            box = ttk.Frame(sliders)
            box.grid(row=0, column=i, sticky="ew", padx=5)
            ttk.Label(box, text=name).pack(anchor="w")
            ttk.Scale(box, from_=lo, to=hi, variable=var, orient=tk.HORIZONTAL).pack(fill=tk.X)
            ttk.Label(box, textvariable=var, width=7).pack(anchor="e")
            sliders.grid_columnconfigure(i, weight=1)

        main = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        main.pack(fill=tk.BOTH, expand=True)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)

        left = ttk.LabelFrame(main, text="Seed image", padding=6)
        right = ttk.LabelFrame(main, text="Iterated image", padding=6)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        right.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        self.seed_label = tk.Label(left, bg="#06090d")
        self.seed_label.pack(fill=tk.BOTH, expand=True)
        self.current_label = tk.Label(right, bg="#06090d")
        self.current_label.pack(fill=tk.BOTH, expand=True)

        lower = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        lower.pack(fill=tk.X)
        self.gain_canvas = tk.Canvas(
            lower,
            height=110,
            bg="#071016",
            highlightthickness=1,
            highlightbackground="#1c3540",
        )
        self.gain_canvas.pack(fill=tk.X)
        self.metrics = ttk.Label(lower, text="step 0", font=("Consolas", 10))
        self.metrics.pack(anchor="w", pady=(5, 0))

    def _load_engine_async(self):
        def worker():
            try:
                self.result_queue.put(("engine", PulsingMAE(self.model_name, self.device)))
            except Exception as exc:
                tb = traceback.format_exc()
                print("\n=== Pulsing Transformer Dream: model initialization failed ===")
                print(tb)
                self.result_queue.put(("engine_error", f"{exc!r}\n\n{tb}"))

        threading.Thread(target=worker, daemon=True).start()

    def load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff"),
                ("All files", "*.*"),
            ]
        )
        if not path:
            return

        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return

        self.seed_image = image.copy()
        self.current_image = image.copy()
        self.step_index = 0
        self.seed_signature = None
        self.current_signature = None
        self._show_images()
        self.status.config(text=f"Loaded {Path(path).name}", foreground="#70f0d2")

    def save_current(self):
        if self.current_image is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")],
        )
        if path:
            self.current_image.save(path)

    def reset_image(self):
        self.running = False
        self.start_btn.config(text="START")
        if self.seed_image is not None:
            self.current_image = self.seed_image.copy()
        self.step_index = 0
        self.seed_signature = None
        self.current_signature = None
        self._show_images()
        self._draw_gains([1.0] * (len(self.engine.layers) if self.engine else 12))
        self._update_metrics(None, None)

    def toggle_run(self):
        self.running = not self.running
        self.start_btn.config(text="PAUSE" if self.running else "START")
        if self.running:
            self.step_once()

    def _gains_for_step(self, step: int):
        n = len(self.engine.layers)
        base = float(self.base_var.get())
        amp = float(self.amp_var.get())
        period = max(1e-3, float(self.period_var.get()))
        depth_cycles = float(self.depth_var.get())
        mode = self.mode_var.get()
        gains = []

        for l in range(n):
            x = l / max(1, n - 1)

            if mode == "flat":
                g = 1.0
            elif mode == "middle":
                bell = math.exp(-0.5 * ((x - 0.5) / 0.18) ** 2)
                g = base + amp * bell
            elif mode == "alternate":
                g = base + amp * (1.0 if (l + step) % 2 == 0 else -1.0)
            elif mode == "strobe":
                phase = (step + l * 0.35) % max(2.0, period)
                g = base + (amp if phase < 1.0 else -0.65 * amp)
            else:
                phase = 2.0 * math.pi * (step / period - depth_cycles * x)
                g = base + amp * math.sin(phase)

            gains.append(float(np.clip(g, -1.0, 3.0)))

        return gains

    def step_once(self):
        if self.busy or self.engine is None or self.current_image is None:
            return

        self.busy = True
        self.step_btn.config(state=tk.DISABLED)
        gains = self._gains_for_step(self.step_index)
        self.engine.set_gains(gains)
        self._draw_gains(gains)

        current = self.current_image.copy()
        blend = float(self.blend_var.get())
        mask = float(self.mask_var.get())
        pixel_noise = float(self.noise_var.get())
        step = self.step_index

        def worker():
            try:
                recon, signature = self.engine.reconstruct(
                    current,
                    mask_ratio=mask,
                    seed=42000 + step,
                )
                cur = np.asarray(
                    current.resize(recon.size, Image.Resampling.LANCZOS),
                    dtype=np.float32,
                ) / 255.0
                rec = np.asarray(recon, dtype=np.float32) / 255.0

                nxt = (1.0 - blend) * cur + blend * rec
                if pixel_noise > 0:
                    rng = np.random.default_rng(9000 + step)
                    nxt += pixel_noise * rng.standard_normal(nxt.shape).astype(np.float32)

                nxt = np.clip(nxt, 0.0, 1.0)
                out_img = Image.fromarray(
                    (nxt * 255.0 + 0.5).astype(np.uint8),
                    mode="RGB",
                )
                pixel_drift = float(np.mean(np.abs(nxt - cur)))
                self.result_queue.put(
                    ("step", out_img, signature, pixel_drift)
                )
            except Exception as exc:
                self.result_queue.put(("step_error", repr(exc)))

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _cosine(a, b):
        if a is None or b is None or a.size != b.size:
            return None
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        if den < 1e-12:
            return None
        return float(np.dot(a, b) / den)

    def _poll(self):
        try:
            while True:
                msg = self.result_queue.get_nowait()
                kind = msg[0]

                if kind == "engine":
                    self.engine = msg[1]
                    self.start_btn.config(state=tk.NORMAL)
                    self.step_btn.config(state=tk.NORMAL)
                    self.status.config(
                        text=f"Ready · {len(self.engine.layers)} pulsing encoder blocks · {self.engine.layer_layout} · {self.device}",
                        foreground="#70f0d2",
                    )
                    self._draw_gains([1.0] * len(self.engine.layers))

                elif kind == "engine_error":
                    self.status.config(text="Model load failed", foreground="#ff6f91")
                    messagebox.showerror("Transformer load failed", msg[1])

                elif kind == "step":
                    _, out_img, signature, pixel_drift = msg
                    self.current_image = out_img
                    self.current_signature = signature
                    if self.seed_signature is None:
                        self.seed_signature = signature.copy()

                    self.step_index += 1
                    self.busy = False
                    self.step_btn.config(state=tk.NORMAL)
                    self._show_images()
                    self._update_metrics(
                        pixel_drift,
                        self._cosine(self.seed_signature, self.current_signature),
                    )

                    if self.running:
                        self.root.after(20, self.step_once)

                elif kind == "step_error":
                    self.busy = False
                    self.running = False
                    self.start_btn.config(text="START")
                    self.step_btn.config(state=tk.NORMAL)
                    self.status.config(text="Iteration failed", foreground="#ff6f91")
                    messagebox.showerror("Iteration failed", msg[1])

        except queue.Empty:
            pass

        self.root.after(80, self._poll)

    def _fit_image(self, image, label, max_size=(610, 520)):
        if image is None:
            label.config(image="")
            return
        im = image.copy()
        im.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(im)
        label.config(image=photo)
        label.image = photo

    def _show_images(self):
        if self.seed_image is not None:
            self._fit_image(self.seed_image, self.seed_label)
        if self.current_image is not None:
            self._fit_image(self.current_image, self.current_label)

    def _draw_gains(self, gains):
        c = self.gain_canvas
        c.delete("all")
        c.update_idletasks()
        w = max(400, c.winfo_width())
        h = max(90, c.winfo_height())
        n = len(gains)
        if n == 0:
            return

        bar_w = w / n
        lo, hi = -1.0, 3.0
        y_zero = h - ((0.0 - lo) / (hi - lo)) * h
        y_one = h - ((1.0 - lo) / (hi - lo)) * h

        c.create_line(0, y_zero, w, y_zero, fill="#5b333c")
        c.create_line(0, y_one, w, y_one, fill="#2f6e67", dash=(3, 3))

        for i, g in enumerate(gains):
            y = h - ((g - lo) / (hi - lo)) * h
            x0 = i * bar_w + 2
            x1 = (i + 1) * bar_w - 2
            fill = "#70f0d2" if g >= 1 else "#58a8d8" if g >= 0 else "#ff6f91"
            c.create_rectangle(x0, y, x1, y_zero, fill=fill, outline="")
            c.create_text(
                (x0 + x1) / 2,
                h - 7,
                text=str(i),
                fill="#8099a3",
                font=("Consolas", 8),
            )

        c.create_text(
            6,
            7,
            anchor="nw",
            text=f"mode={self.mode_var.get()}   g range {min(gains):.2f} … {max(gains):.2f}",
            fill="#b9ccd4",
            font=("Consolas", 9),
        )

    def _update_metrics(self, pixel_drift, semantic_cos):
        parts = [
            f"step {self.step_index}",
            f"mode={self.mode_var.get()}",
        ]
        if pixel_drift is not None:
            parts.append(f"pixel Δ={pixel_drift:.4f}")
        if semantic_cos is not None:
            parts.append(f"encoder cosine vs first={semantic_cos:.4f}")
        if self.engine is not None:
            parts.append(f"mask={self.mask_var.get():.2f}")

        self.metrics.config(text="   |   ".join(parts))


def choose_device(requested: str):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(
        description="Iterate an image through a frozen ViT-MAE while transformer depth gains pulse."
    )
    ap.add_argument("--model", default="facebook/vit-mae-base")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    args = ap.parse_args()

    device = choose_device(args.device)
    root = tk.Tk()
    app = DreamApp(
        root,
        model_name=args.model,
        device=device,
    )
    root.mainloop()

    if app.engine is not None:
        app.engine.close()


if __name__ == "__main__":
    main()
