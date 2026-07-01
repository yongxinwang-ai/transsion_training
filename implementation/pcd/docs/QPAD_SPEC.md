# QPAD Spec

## Goal
Add a minimal structural path that helps the model read the relocated question panel without overwriting the base model's original-format behavior.

## Input views
### Prompt-in-image inference
- text: `Help me solve the problem`
- image 1: full prompt-in-image composite
- image 2: deterministic crop of the question panel

### Original inference
- unchanged
- no panel crop path active

## Backbone policy
- freeze language model
- freeze base vision tower
- freeze base multimodal projector in the first version

## New trainable components
### 1. Panel token reducer
Input:
- crop-image visual tokens from the frozen vision tower

Operation:
- small MLP or Perceiver-style reduction from many crop tokens to `K` panel summary tokens

Recommended first version:
- linear -> GELU -> linear
- output `K=16` summary tokens

### 2. Panel fusion adapter
Input:
- decoder hidden states
- `K` panel summary tokens

Operation:
- one or two layers of cross-attention from decoder hidden states to panel summary tokens
- residual add with learned scalar gate

Recommended first version:
- single cross-attention block
- one FFN block
- scalar gate initialized near zero

### 3. View gate
- activate the adapter only for prompt-in-image samples
- bypass completely for original-view samples

## Forward sketch
1. Encode full composite image with the frozen base path.
2. Encode panel crop with the same frozen vision tower.
3. Reduce crop tokens to panel summary tokens.
4. During decoding, inject panel summary tokens through the fusion adapter.
5. Predict canonical final answer.

## Losses
### Main
- `L_ans`: CE on canonical final answer
- `L_distill`: KL from frozen original-view teacher on answer tokens

### Optional
- `L_anchor`: weak original-view anchor for sanity

## First ablation set
1. frozen backbone only
2. QPAD without distillation
3. QPAD with distillation

## Why this insertion point
- decoder-side cross-attention is simpler than editing the vision tower
- keeps the added module small
- preserves base multimodal path
- easy to disable cleanly on original-format inputs

## Failure signals to watch
- no gain over frozen backbone: crop fusion too weak or crop tokens too noisy
- original drop despite frozen backbone: adapter leakage path too strong
- MathVision no gain: layout bottleneck broader than panel reading alone

## Recommended first benchmark
- MathVista prompt-in-image

This is the clearest place where current evidence suggests underuse of the relocated question panel.
