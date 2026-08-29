# Pulsing Transformer Dream

A Saturday-night misuse experiment.

> Start with an image. Run it through a frozen pretrained image transformer. Reconstruct the image. Feed that reconstruction back into the same transformer. Repeat — while transformer depth itself pulses.

This is not a diffusion model in the technical generative-model sense. It is an iterative stochastic image-to-image dynamical system built from a pretrained ViT-MAE.

## The intervention

For encoder block l, ordinary execution is

    h_(l+1) = B_l(h_l)

The demo replaces it at inference time with

    h_(l+1) = h_l + g_l(t) * [B_l(h_l) - h_l]

So:

    g = 1.0   original pretrained block
    g = 0.0   bypass block
    g = 0.5   half-strength learned update
    g > 1.0   extrapolate the block update
    g < 0.0   push opposite the learned update

No weights are trained.

The default WAVE schedule creates a traveling gain wave through transformer depth.

The image recurrence is approximately:

    next_image = (1 - blend) * current + blend * MAE_reconstruction(current)

Optional MAE masking and pixel noise make the trajectory stochastic.

## Run on Windows

    run_pulsing_dream.bat

Or manually:

    pip install -r requirements_dream.txt
    python experiments/pulsing_vit_mae_dream.py

The first run downloads facebook/vit-mae-base from Hugging Face.

CUDA is used automatically when available. CPU works, but each image iteration can be slow.

## Controls

- LOAD IMAGE — choose the seed.
- START / PAUSE / STEP — iterate the feedback loop.
- RESET IMAGE — return to the seed without changing transformer settings.
- ALL PASS — ordinary encoder execution; only repeated reconstruction remains.
- WAVE — traveling sinusoidal layer gains.
- MIDDLE — amplify a bell-shaped middle-depth band.
- ALTERNATE — neighboring layers receive opposite gain perturbations.
- STROBE — short moving pulses through depth.
- base — center gain.
- pulse amp — how hard to perturb execution.
- period — pulse period in image iterations.
- depth cycles — phase change from first to last encoder block.
- image blend — how much reconstruction replaces the current image.
- mask ratio — MAE patch masking per iteration.
- pixel noise — optional external diffusion-like noise.

The lower bars show the actual per-layer gains applied at the current step.

## First things to try

Control — reconstruction drift only:

    ALL PASS
    image blend 0.4
    mask ratio 0.0
    pixel noise 0.0

Gentle semantic wave:

    WAVE
    base 1.0
    pulse amp 0.25
    period 12
    depth cycles 1
    blend 0.35
    mask 0.15

Wrong-machine mode:

    WAVE
    base 0.8
    pulse amp 0.9
    period 8
    depth cycles 1.5
    blend 0.75
    mask 0.35

Brutal:

    STROBE
    base 0.5
    pulse amp 1.5
    blend 0.9
    mask 0.5

Expect garbage. Garbage is part of the map.

## What would actually be interesting

Not merely that the picture becomes psychedelic. Interesting observations would include:

- a stable visual attractor under one depth schedule;
- a repeating cycle rather than monotonic collapse;
- two schedules producing different stable transformations from the same seed;
- a sharp transition when one depth band crosses a gain threshold;
- a feature disappearing and later reappearing as the gain wave moves;
- semantic drift much larger than pixel drift, or vice versa;
- different seeds converging toward a schedule-specific visual family.

The UI reports pixel change per iteration and cosine similarity of the final encoder state to the first iteration as crude receipts.

## Honesty

ViT-MAE was not trained to be iterated like this, and it was not trained with pulsing depth gains.

Therefore the default expectation is degradation.

> What dynamics appear when a frozen learned visual transform is repeatedly forced to consume its own reconstruction while its normal depth execution is periodically broken?
