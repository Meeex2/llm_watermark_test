Let me explain the training and sampling process in the current implementation:

1. **Training Process**:
- The training is still done using the original EDM (Elucidating the Design Space of Diffusion-Based Generative Models) loss, not DDIM
- This is because DDIM is primarily a sampling strategy, not a training method
- The model learns to predict noise during training using the EDM loss, which is a well-established approach for diffusion models

2. **DDIM Usage**:
- DDIM is used only during the sampling/generation phase
- It's implemented in the `sample_ddim` method of the Model class
- DDIM provides a more efficient sampling process compared to the original DDPM sampling

3. **Do you need to retrain?**
- No, you don't need to retrain if you already have a trained model
- The existing model weights will work with the new DDIM sampling and DPM-Solver
- This is because the underlying noise prediction network remains the same

Here's how the process works:

```python
# During training (existing process):
model.train()
loss = model(inputs)  # Uses EDM loss for training

# During generation (new process):
model.eval()
samples = model.sample_ddim(num_samples, in_dim, num_steps=50, eta=0.0)  # Uses DDIM for sampling
```

The key benefits of this approach are:
1. Maintains the proven training stability of EDM
2. Gains the sampling efficiency of DDIM
3. Adds the noise recovery capability of DPM-Solver

If you want to generate samples with the new implementation, you can simply use:
```bash
python main.py --mode generate --dataname adult --gpu 0 --num_samples 1000
```

The model will use DDIM for sampling, which should be faster and potentially produce better quality samples than the original sampling method.
