# Quick Start: GPU Validation & Oracle Timeout

## What Changed?

### 1. GPU Validation Tool (NEW)
- **Before:** Agent checked GPU memory too early, causing false failures
- **After:** Timing-based validation that actually works

### 2. Oracle Timeout (NEW)
- **Before:** Oracle could hang forever
- **After:** 10-minute timeout per call with streaming progress

---

## How to Use

### GPU Validation

**Agent will automatically use this on first turn:**
```
GPUValidate(framework='pytorch', model_size='small', batch_size=256)
```

**Expected output:**
```
✅ GPU validation PASSED for pytorch
Performance matches GPU expectations (1.2s < 12s threshold)
GPU training is working correctly. Proceed with full training.
```

**If fails:**
```
❌ GPU validation FAILED for pytorch
CPU fallback detected. Check your training code:
- PyTorch: Ensure model and data use .to(device) or .cuda()
```

### Oracle Consultation

**Same usage as before:**
```python
Oracle(query="Review my training strategy for this competition")
```

**New progress output:**
```
🔮 Oracle: Consulting O3 and DeepSeek-R1 in parallel...
🔮 O3 streaming... ........ ✓
🔮 DeepSeek-R1 streaming... .......... ✓
🔮 Oracle: O3 Critic synthesizing optimal plan...
🔮 O3 Critic synthesizing... ............. ✓
```

**If timeout:**
```
ERROR: O3 timed out after 10 minutes - returning partial response if available
```

---

## Key Differences

### GPU Detection: Timing > Memory

**Old method (unreliable):**
- Checked GPU memory after first batch
- <10% memory = error (false positive!)

**New method (reliable):**
- Benchmark 100 batches with timing
- >5s for 100 batches = CPU (10-20x slower than GPU)

**During training:**
- Prints epoch timing: `Epoch 0 completed in 35.2s`
- GPU: 30-60s/epoch for EfficientNet-B3
- CPU: 300-600s/epoch for EfficientNet-B3

### Oracle: Timeout + Streaming

**Old method:**
- Silent wait
- Could hang forever

**New method:**
- Streaming progress dots
- 10-minute timeout per call
- Returns error if times out

---

## Testing

### Test GPU Validation (Manual)

```bash
cd /Users/Yifan/canada-research/mle-bench/environment
python validate_gpu.py --framework pytorch --model-size small --batch-size 256
```

Expected: `✅ GPU TRAINING CONFIRMED` in ~2-3 seconds

### Test in Competition (Automatic)

1. Run any competition
2. Agent will call GPUValidate on first turn
3. Check output shows "GPU validation: PASSED"

### Test Oracle (Automatic)

1. Call Oracle during competition
2. Watch for streaming progress: `🔮 O3 streaming... ........ ✓`
3. Verify completes within 10 minutes

---

## Troubleshooting

### GPU Validation Fails

**Symptom:** `❌ GPU validation FAILED`

**Check:**
1. Is CUDA available? `nvidia-smi`
2. Is PyTorch using GPU? `python -c "import torch; print(torch.cuda.is_available())"`
3. Is model on GPU? Check training code uses `.to(device)`

**Fix:**
- Ensure Docker container has `--gpus all` flag
- Check CUDA drivers installed
- Verify training code moves model and data to GPU

### Oracle Times Out

**Symptom:** `ERROR: O3 timed out after 10 minutes`

**This is normal if:**
- Query is very complex
- API is slow
- Network issues

**Agent will:**
- Continue with partial response
- Try using available results
- Suggest simpler query or fallback strategy

### False CPU Detection

**Symptom:** Agent thinks GPU training but it's actually CPU

**Check epoch timing:**
- GPU: <2 min/epoch for medium models
- CPU: >5 min/epoch for medium models

**If >5 min/epoch → KILL TRAINING**
- Not using GPU correctly
- Fix training code and restart

---

## Quick Reference

### GPU Speed (A100 40GB)
| Model | GPU Time/Epoch | CPU Time/Epoch |
|-------|---------------|----------------|
| EfficientNet-B3 | 30-60s | 600-1200s |
| ResNet-50 | 20-30s | 300-600s |
| EfficientNet-B4 | 60-90s | 900-1800s |

**Rule:** If epoch >5 min → CPU (KILL IMMEDIATELY)

### Oracle Timeout
- O3: 10 minutes
- DeepSeek-R1: 10 minutes
- O3 Critic: 10 minutes
- **Total max:** ~30 minutes

### Progress Indicators
- GPU validation: `✅ GPU TRAINING CONFIRMED`
- Oracle streaming: `🔮 O3 streaming... ........ ✓`
- Training epoch: `Epoch 0 completed in 35.2s`

---

## Files Changed

**Created:**
- `mle-bench/environment/validate_gpu.py` - GPU validation script
- `mle-bench/agents/agent_v5_kaggle/tools/gpu_validate.py` - GPUValidate tool

**Modified:**
- `mle-bench/agents/agent_v5_kaggle/kaggle_agent.py` - Added GPUValidate tool
- `mle-bench/environment/training_hints.txt` - Fixed GPU validation code
- `agent_v5/tools/oracle.py` - Added timeout + streaming

**Documentation:**
- `GPU_VALIDATION_FIX.md` - Complete GPU fix details
- `ORACLE_TIMEOUT_STREAMING.md` - Complete Oracle timeout details
- `SESSION_SUMMARY.md` - Full session summary
- `QUICK_START.md` - This guide

---

## Next Competition Run

**Expected improvements:**
1. GPU validation on first turn (no false failures)
2. Correct GPU detection during training (no CPU fallback)
3. Oracle streaming progress (no indefinite waits)
4. Clear epoch timing prints (easy to spot CPU training)

**Watch for:**
- `✅ GPU validation PASSED` on first turn
- `Epoch 0 completed in XX.Xs` during training (should be <120s for medium models)
- `🔮 O3 streaming...` when Oracle called
- No hanging on Oracle calls (max 30 min total)

---

## Summary

✅ GPU validation is now **timing-based** (reliable)
✅ Oracle has **10-minute timeout** (no hanging)
✅ Agent gives **clear progress feedback** (streaming)
✅ Training shows **epoch timing** (easy GPU detection)
✅ No more false positives from **premature memory checks**

**Ready to use immediately** - no configuration needed!
