# Experimental moon composite

This opt-in mode takes a short exposure immediately **before** each normal night
exposure while indi-allsky's existing moon mode is active. It replaces the
bright lunar silhouette in the display image using the short exposure, with a
small feathered edge. This is replacement/alpha compositing, not a screen blend
(which would keep an already white moon white).

## Configuration

Currently supported only by `CAMERA_INTERFACE: "indi"` (native active INDI).
Disabled by default. In **Camera Settings > Gain & Binning Modes > Moon Mode
Profile**, enter a valid Moon Mode gain (zero is valid) to reveal **Recover Moon
Surface Detail (Experimental)**. The option appears only for native INDI cameras.
Turn it on to reveal the short exposure, four search-region coordinates, and
mask threshold. Save and restart capture. The option is also shown on page load
when a valid gain is already saved; entering a gain never enables it automatically.

Clearing/invalidating the gain or selecting another camera interface hides and
unchecks the option. Saved tuning values are retained. The same settings remain
available through the configuration CLI's `edit` or export/edit/import workflow.

Add this object to the configuration:

```json
"MOON_COMPOSITE": {
    "ENABLE": true,
    "EXPOSURE": 0.001,
    "ROI": [1000, 800, 1200, 1000],
    "THRESHOLD": 40
}
```

- `EXPOSURE`: short exposure in seconds. Must be within the camera's hardware
  limits; it is independent of the normal sky exposure minimum. Start at 1 ms,
  then tune for your camera/lens and moon-mode gain. If the moon is still clipped,
  reduce it. The short frame uses the **same gain and binning** as the sky frame.
- `ROI`: `[left, top, right, bottom]` in **binned sensor pixels, before any
  rotation, flipping, or crop**. All coordinates must be even to preserve Bayer
  phase. Keep this region small, but allow for lunar movement and at least four
  pixels of margin around the moon. Update it as the moon moves; this prototype
  does not track the moon across the sky.
- `THRESHOLD`: lunar silhouette threshold in 8-bit sensor-scaled short-frame
  intensity (1-249). Default 40. Lower it if dark lunar regions are missed.

Normal moon-mode altitude/phase gates still apply. Daytime, focus, synchronous
test captures, SQM captures, and sky exposures no longer than the short exposure
are not bracketed. Existing moon-mode exposure/gain behavior is unchanged.
Both capture and image processing require active night-time Moon Mode. Outside
it there are no short exposures, additional BLOB waits, short-frame reads,
debayering or compositing. If Moon Mode ends before a pair is processed, the short
frame is discarded and the long image is processed normally.

## Safety and output

No extra image job is queued: the short frame travels with its corresponding
long frame. The compositor rejects small targets, multiple significant bright
objects, targets touching the region edge, clipped short exposures, or mismatched
geometry/settings, with an explicit log message. The long image remains unchanged
if replacement cannot be made. Temporary short frames are removed after
processing (including failed long-frame processing), abort, or orderly shutdown.

Only the display image is composited, after sky stretch and exposure/star/meteor
measurements but before rotation/crop. FITS, raw exports, sky exposure control,
and stacking inputs remain long-exposure originals. Subsequent white balance,
gamma and contrast adjustments also affect the replacement moon.

## Cost and limitations

One camera cannot take a short exposure *inside* its ongoing long exposure.
There is always an extra exposure **and full sensor readout/transfer**. Existing
cadence is retained when the pair fits within the exposure period; otherwise
capture slows. This mode cannot promise zero capture overhead.

Image processing reads/debayers only the short-frame ROI and blends that ROI,
not a second full image pipeline. Timing is logged as `Moon composite applied=...`.
Disabled mode performs no extra capture, FITS read, or pixel processing.

A local Windows benchmark with a 3008x3008, 16-bit Bayer FITS input measured
15 ms median / 20 ms p95 for a 200x200 ROI, and 52 ms / 65 ms for a 512x512 ROI
(100 iterations, warm filesystem cache). These include FITS open/ROI read,
debayer, scaling, mask generation and blend, but **exclude camera acquisition
and transfer**. Raspberry Pi and cold-storage timings remain unverified.

This is a deliberately small first implementation, not HDR recovery of the whole
scene: long-exposure glow/bloom outside the lunar mask remains. There is no
inter-frame registration or lunar-motion compensation; long exposures, vibration,
cloud motion, and a low-resolution lunar disk can make the result imperfect.
The short exposure is not dark-calibrated or separately stretched. It cannot
invent surface detail that the lens/sensor does not resolve. A cloud or terrestrial
light in the configured ROI can still resemble the moon: use a tight unobstructed
ROI and inspect the result before enabling unattended.

Validate with real camera frames before relying on it. Synthetic tests exercise
pairing, skip paths, cleanup, monochrome/RGB FITS and 8/12/16-bit scaling, but do
not substitute for camera-driver and night-sky testing.
