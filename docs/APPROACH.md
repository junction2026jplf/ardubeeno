# How we built it: first the mite check, then the bee finder

ArduBeeno uses two kinds of models: a **mite check** that looks at a close-up of one bee, and a **bee finder** that
spots the bees in the camera picture and cuts out those close-ups. We built them in that order. Details:
[TRAINING.md](TRAINING.md), numbers: [RESULTS.md](RESULTS.md).

## Step 1: the mite check, on close-ups of single bees

We started with the [VarroaDataset](https://zenodo.org/records/4085044): 13,509 pictures of **one bee each**
(160x280, bee upright), with the mites marked. A mite covers only a few pixels in a full camera picture, but in a
close-up it is clearly visible, and the dataset answers exactly one question: mite or not.

We trained a fast **classifier** (a mite score for the whole bee) and a **mite detector** (finds the mite itself).
The classifier looks first; only when it is unsure does the detector also look, and the two decide together.

## Step 2: the bee finder, which makes the close-ups

A camera at the hive sees the whole entrance, with several bees at any position and angle. So we trained a second
model with one class, "bee": first on public hive and landing-board pictures, then on our own camera recordings.
Later versions added synthetic bees at every angle and pictures without bees, so it also learns what is *not* a bee.

`crop_bee` (in `applab/ardubeeno/python/bees.py`) turns each bee box into a close-up like the training pictures:
portrait shape, turned upright, scaled to the right size. So the mite check from step 1 works unchanged. A tracker
follows each bee, so the board can look at the same bee several times before deciding.

## Closing the loop

Through the camera, the mite check dropped from F1 0.901 to 0.725 (blur, moiré, colour shifts). We filmed the
dataset through the camera, cut the bees out with `crop_bee` and fine-tuned the mite check on them. It now finds
about 93 of every 100 bees with a mite. The same steps are the plan for real hive footage.

## Why two models

- The right data for each part: a mite dataset of close-ups, bee datasets of whole scenes.
- The close-up makes the mite big enough to see.
- Each model does one simple thing, so both run on the board's CPU.
- Each part can be improved without retraining the other.
