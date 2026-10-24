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
long frame. One locked capture record owns the bracket's local ID, requested
long exposure/gain/binning, short frame, phase and completion status. The capture
loop asks that operation to advance; it no longer coordinates separate pending
settings and short-frame flags. Only one operation may be outstanding.
Each exposure request accepts at most one BLOB. Callbacks received when no image
is awaited are ignored; that does not identify a duplicate arriving during a
later request. A reported FITS `EXPTIME` that differs from the request
is rejected rather than relabelled as the other half of a bracket. The tolerance
is 1% or one microsecond, whichever is larger, to allow driver rounding.
Local operation state, callback ownership and camera readiness coordinate the
pair; metadata is a defensive cross-check, not a camera-generated frame ID.
When `DATE-OBS` is supplied, it must be parseable and not predate the request by
more than one second (allowing whole-second timestamp precision). Offset-free
FITS timestamps are interpreted as UTC. Keep camera and host clocks synchronized.

**Driver limitation: stale-frame rejection is not guaranteed across INDI
drivers.** `EXPTIME` is an exposure duration, not a unique frame identifier.
If a driver omits `DATE-OBS`, a delayed short frame from an earlier bracket with
the same exposure duration can pass the checks while a new short frame is
awaited. It can receive the new local bracket ID and be paired with the new long
frame; the ID does not prove when the camera captured it. Coarse, repeated or
incorrect driver timestamps can also limit detection even when the field exists.
Locking, readiness checks and connection resets do not establish a universal
camera-buffer/frame-identity guarantee. Driver-specific capture evidence is
still required; missing timestamps are currently accepted, not automatically
treated as a reason to suspend compositing.

Rejected BLOBs leave the request waiting for a valid BLOB or the existing timeout.
A timed-out bracket, or missing `EXPTIME` needed to verify the pair, suspends
compositing until configuration reload/restart, without changing the saved
enable setting. The incomplete pair is discarded, the capture connection is
reset, and the camera must report ready before an ordinary long exposure is
requested. The worker continues normal sky captures, even if that driver's
ordinary images omit or round `EXPTIME`; it does not repeatedly retry a failing
bracket. If reconnection/idle verification fails, the normal worker error path
reports the failure instead of pretending that a fallback image was captured.
Check the requested/reported values in the `EXPTIME` warning and the driver's
exposure limits/step size for **both** short and long captures before re-enabling.
The strict pairing tolerance is not automatically relaxed. A frame that fails
these metadata checks is not composited; passing them does not prove its identity.
Malformed FITS headers and truncated data are rejected with a diagnostic while
waiting for a replacement frame or timeout; partial temporary files are removed.

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

**16-bit night contrast compatibility:** when both `CONTRAST_ENHANCE_16BIT`
and `NIGHT_CONTRAST_ENHANCE` are enabled, Moon replacement is skipped with an
actionable warning. This local CLAHE step runs before replacement and cannot be
reproduced by the shared global stretch lookup. No short exposure or Moon
transfer table is generated; normal long-exposure processing is unchanged.
Any already-queued pair is also skipped before reading the short ROI, with normal
temporary-file cleanup. The saved Moon enable setting is retained, not suspended
as a capture failure. Disable `CONTRAST_ENHANCE_16BIT` to allow replacement;
ordinary 8-bit contrast enhancement runs after blending and remains supported.
If night contrast enhancement is off, the 16-bit option alone does not block
replacement.

When sky stretching is enabled for Moon Mode, the short ROI reuses the **actual
lookup tables from that long frame**, at native bit depth before conversion to
8-bit display pixels. This includes Mode 1 gamma/levels, Mode 2 MTF (including
the two-pass variant) and Mode 3 adaptive MTF. Adaptive parameters are not fitted
again to the short ROI. Threshold selection, geometry and short-exposure clipping
checks still use the original linearly scaled ROI, so brightening the display
does not enlarge the lunar mask or falsely report sensor saturation.
Stretch-disabled Moon Mode remains linear. Split-screen preview uses the matching
transfer on each side and preserves the central divider.

Using the same transfer fixes the linear-versus-stretched processing mismatch;
it does **not** guarantee equal brightness from two different exposures or a
natural boundary. The incompatible pre-blend 16-bit CLAHE combination is skipped,
not independently fitted to the short ROI. Denoising and dark calibration are not
replayed on the short ROI. Real pairs must still be inspected for residual discontinuities;
no independent auto-stretch or automatic brightness matching is assumed.

## Cost and limitations

One camera cannot take a short exposure *inside* its ongoing long exposure.
There is always an extra exposure **and full sensor readout/transfer**. Existing
cadence is retained when the pair fits within the exposure period; otherwise
capture slows. This mode cannot promise zero capture overhead.

Image processing reads/debayers only the short-frame ROI and blends that ROI,
not a second full image pipeline. FITS reads use `hdu.section` with
`memmap=False`: section access reads only the requested pixels, without loading
the full image. Explicit memory mapping is not used because Astropy rejects it
for unsigned 16-bit FITS images with `BZERO` scaling. The FITS file stays open
through section extraction and decoding, and closes before short-frame cleanup.
Blending reuses two ROI-sized float buffers for weighted pixels, addition and
rounding, preserving the original float32 arithmetic and round-to-even output.
Clipping, connected-component ambiguity checks and alpha feathering are unchanged.
Timing is logged as
`Moon composite bracket=<id> applied=... in ...s`.
Disabled mode performs no extra capture, FITS read, or pixel processing.
The sky transfer is combined into one 8-bit display lookup table per frame;
the ROI needs one lookup, not another full-frame stretch or adaptive-statistics
calculation. Tables are prepared for replay only when compositing and Moon Mode
stretching are active **and that image has a paired short frame**, not for ordinary
fallback captures.

A local Windows benchmark with a 3008x3008, 16-bit Bayer FITS input measured
15 ms median / 20 ms p95 for a 200x200 ROI, and 52 ms / 65 ms for a 512x512 ROI
(100 iterations, warm filesystem cache). These include FITS open/ROI read,
debayer, scaling, mask generation and blend, but **exclude camera acquisition
and transfer**. Raspberry Pi and cold-storage timings remain unverified.

For the added display-transfer step, a separate 100-iteration warm Windows test
with 16-bit RGB ROIs measured about 0.24 ms median for the 200x200 lookup and
1.7 ms for 512x512, plus 0.2-0.7 ms to prepare the combined curve per frame.
Those are lookup/curve costs only, not total compositor or acquisition timings;
embedded-host performance remains unverified.

A separate synthetic Windows comparison of the complete `blend()` call (60
interleaved iterations, NumPy 2.4.2 / OpenCV 4.13.0) measured 512x512 mono medians
of 5.44 ms before / 3.89 ms with buffer reuse, and BGR medians of 26.01 / 25.24 ms.
Peak Python-tracked allocations fell from 5.50 to 4.54 MiB for mono and 12.00 to
9.07 MiB for BGR; these are not process RSS or native OpenCV allocation totals.
Outputs matched exactly. These blend-only measurements exclude FITS I/O and
camera acquisition, and do not establish an embedded-host speedup.

This is a deliberately small first implementation, not HDR recovery of the whole
scene: long-exposure glow/bloom outside the lunar mask remains. There is no
inter-frame registration or lunar-motion compensation; long exposures, vibration,
cloud motion, and a low-resolution lunar disk can make the result imperfect.
The short exposure is not dark-calibrated or independently auto-stretched. It cannot
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
locked bracket transitions, missing/malformed exposure metadata, structurally
invalid FITS headers/truncated pixels, readout geometry and clipped short frames.
Injected persistent short- and long-exposure metadata mismatches exercise
timeout/disconnect/cleanup, a freshly constructed camera client, an idle-control
check and continued ordinary captures within the same worker. Tests also check
that a permanently busy camera surfaces a recovery error, and that a new worker
can resume pairing with corrected metadata or run with the feature disabled.
These simulate recovery; they do not certify any real driver's exposure reporting.
Transfer tests cover all four stretch variants, mono/RGB and 8/12/16-bit data,
verify exact agreement with the sky transfer, preserve the linear mask/clipping
checks, and ensure skipped stretches cannot reuse a previous frame's transfer.
Compatibility tests verify that active 16-bit night CLAHE suppresses brackets
and transfer preparation, and that queued pairs leave the CLAHE-processed mono/RGB
sky and original FITS unchanged without reading the short frame.
A timestamp characterization test replays the same old short BLOB into a later
bracket: without `DATE-OBS` it is accepted, while an accurately reported old
timestamp is rejected. This intentionally demonstrates the identification limit,
not guaranteed stale-frame protection.
The INDI transport, clock, database and operating-system signal registration are
explicit test doubles. These tests do **not** start an INDI server, launch the
worker supervisor, use a real camera, or exercise the entire Flask/database
application lifecycle.

The Python configuration test invokes the Node.js built-in test runner. Install
Node.js (20+ recommended) and put `node` on `PATH` before running
`python -m pytest tests/image tests/flask/test_moon_composite_config.py -q`.
The test fails explicitly if Node is missing; it does not silently skip UI
coverage. Existing repository workflows do not run this pytest suite or guarantee
Node for it, so any added test job must provision Node explicitly. Node is a test
prerequisite here, not a new application dependency.

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
   limit detection of delayed frames with the same exposure; metadata/log IDs
   alone cannot prove freshness. Record driver-specific evidence, using frame
   content or driver traces where available, rather than claiming universal
   stale-frame rejection.
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
   acquisition resumes after the connection reset, without reusing stale frames
   or retrying brackets until reload/restart. Check actual driver `EXPTIME`
   rounding/omission for both exposures, abort support and busy-to-idle timing.
   If recovery cannot reach idle, verify that the error is reported rather than
   accepting an uncertain frame. Reload after correcting the cause and confirm
   that verified pairing resumes.

Repeat with a second camera/driver before making cross-driver reliability
claims. Keep the feature experimental until the evidence above is reviewed.
