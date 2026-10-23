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
long frame. The camera accepts one BLOB per requested exposure. Duplicate
callbacks are ignored, and a reported FITS `EXPTIME` that differs from the request
is rejected rather than relabelled as the other half of a bracket. The tolerance
is 1% or one microsecond, whichever is larger, to allow driver rounding.
When `DATE-OBS` is supplied, it must be parseable and not predate the request by
more than one second (allowing whole-second timestamp precision). Offset-free
FITS timestamps are interpreted as UTC. Keep camera and host clocks synchronized.
Rejected BLOBs leave the request waiting for a valid BLOB or the existing timeout;
timeouts disconnect/restart capture rather than reuse an incomplete bracket.
If either frame omits `EXPTIME`, capture continues with the original long image
and logs the driver metadata limitation instead of attempting an unverified blend.

The compositor rejects small targets, multiple significant bright
objects, targets touching the region edge, clipped short exposures, or mismatched
geometry/settings (including Bayer offsets, subframe origin, bit format and row
order), with an explicit log message. The long image remains unchanged
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

## Validation status and real-camera acceptance

**Real-camera validation is outstanding. No captured lunar pairs or hardware
timing logs have been supplied.** A passing synthetic or simulated-transport test
must not be presented as evidence that lunar detail looks convincing or that an
INDI camera/driver maintains cadence.

The existing AST-isolated tests remain. Additional boundary integration tests
import the complete camera, capture and image-processing modules, run their real
constructors and the capture worker's main loop, deliver BLOBs through both
legacy and newer INDI callback entry points, transfer jobs through a real
multiprocessing queue, and decode/composite real FITS files containing synthetic
pixels. They cover delayed/duplicate callbacks, timeout/shutdown cleanup,
missing/malformed exposure metadata, readout geometry and clipped short frames.
The INDI transport, clock, database and operating-system signal registration are
explicit test doubles. These tests do **not** start an INDI server, launch the
worker supervisor, use a real camera, or exercise the entire Flask/database
application lifecycle.

Before claiming hardware readiness, record the following for each tested camera
and driver combination. At least 100 consecutive lunar brackets is a suggested
starting run, not a universal guarantee:

1. **Equipment and settings:** camera model, sensor resolution/bit depth, driver
   name/version, INDI/PyIndi versions, host/USB connection, lens, gain, binning,
   sky/short exposures, ROI, threshold and exposure period. Confirm the lunar
   disk has enough resolved pixels to contain genuine surface detail.
2. **Original evidence:** retain unmodified short and long FITS frames, final
   display images and capture/processing logs for the same run. Short frames are
   temporary in this experiment; use an existing trusted passive BLOB recorder
   if available. Do not run a second exposure controller alongside indi-allsky.
   Separately captured manual pairs can assess optical quality, but cannot prove
   indi-allsky's pairing or callback timing.
3. **Pairing:** correlate `Moon bracket <id> short received`, `queued` and
   `Moon composite bracket=<id>` log entries. Compare actual FITS `EXPTIME` and
   `DATE-OBS` with request times/settings. Check that every accepted long frame
   has exactly one intended short partner and that rejected/missing frames never
   become another bracket's partner. These bracket IDs are **local bookkeeping,
   not IDs supplied by the camera**. Missing or coarse driver timestamps still
   limit detection of delayed frames with the same exposure; hardware evidence
   remains necessary.
4. **Image quality:** examine full-resolution side-by-side long, short and
   composite crops. Confirm surface contrast comes from the actual short frame,
   not clipping, interpolation or the mask edge. Inspect alignment, crescent
   phases, lunar motion, clouds, bloom and colour/brightness discontinuities.
   Record failures as well as successes. A short exposure cannot recover detail
   that the lens and sensor do not resolve.
5. **Cadence:** measure consecutive sky-frame start times, short/long acquisition
   and readout times, queue depth, and compositor median/p95 on the intended host.
   Compare the same setup with the feature disabled. Test both a period with
   enough headroom and one shorter than the combined acquisition/readout time;
   report any slowdown rather than inferring it from the short exposure alone.
6. **Transitions and recovery:** verify inactive Moon Mode/day/focus/SQM do not
   acquire moon brackets. Check Moon Mode ending with a bracket in flight,
   orderly shutdown, and a deliberately interrupted capture in a controlled
   session. Confirm incomplete temporary frames are removed and normal
   acquisition resumes after restart without reusing stale frames.

Repeat with a second camera/driver before making cross-driver reliability
claims. Keep the feature experimental until the evidence above is reviewed.
