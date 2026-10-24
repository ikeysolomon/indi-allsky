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
Turn it on to reveal the short exposure time, the only numeric UI setting.
Moon location, search-region size, and mask threshold are automatic.
Save and restart capture. The option is also shown on page load
when a valid gain is already saved; entering a gain never enables it automatically.

Clearing/invalidating the gain or selecting another camera interface hides and
unchecks the option. Saved tuning values are retained. The same settings remain
available through the configuration CLI's `edit` or export/edit/import workflow.

Add this object to the configuration:

```json
"MOON_COMPOSITE": {
    "ENABLE": true,
    "EXPOSURE": 0.001
}
```

- `EXPOSURE`: short exposure in seconds. Must be within the camera's hardware
  limits; it is independent of the normal sky exposure minimum. Start at 1 ms,
  then tune for your camera/lens and moon-mode gain. If the moon is still clipped,
  reduce it. The short frame uses the **same gain and binning** as the sky frame.
- Optional advanced `SEARCH_MARGIN`: prediction safety margin as a fraction of
  the aligned image-circle diameter, default `0.05`, allowed range `0.01`-`0.1`.
  It is available through configuration export/edit/import, not another UI control,
  and is retained when saving the camera settings. Sensor transforms and binning
  scale this margin; the minimum is 24 sensor pixels.

Old `ROI` and `THRESHOLD` values are ignored and removed on the next configuration
save/import. No manual migration or coordinate updates are needed.

### Automatic localization

The Moon's topocentric altitude/azimuth is calculated for each short exposure
using the camera location and the exposure midpoint. FITS `DATE-OBS` is preferred
(offset-free timestamps mean UTC); otherwise the capture receipt time and elapsed
capture duration estimate that midpoint. Wall-clock processing time is not used.
VirtualSky's lens projection, pointing, offsets and enabled distortion correction
map that direction into image pixels. The image pipeline's rotation, flips,
crop, scale and borders are inverted to recover sensor coordinates. A saved
reference binning is converted to the current binning.

Only a neighbourhood around that predicted position is searched. Its margin
scales with the reference image-circle diameter, with a 24-sensor-pixel minimum.
If no reliable candidate is found, the margin doubles **once**. There are at most
two search windows, never an indefinite expansion or an unguided fallback.
The detector never falls back to unrelated full-frame lights if prediction or
alignment fails. Within this neighbourhood, a max-pooled brightness preview
(at most 768 pixels per dimension) preserves bright seeds without full-frame debayering.
Bayer previews pool complete CFA cells. Starting with the brightest remaining
pixel, the detector grows a local window until the connected bright region is
enclosed, deriving its threshold with Otsu's method and a brightness floor.
It examines at most three bright seeds per window (six total with expansion),
rejecting tiny sources, image-edge
targets, and shapes without a sufficiently circular limb. A partial illuminated
disk can pass; narrow crescents and poorly resolved disks may be skipped.
The apparent lunar diameter from ephemeris is projected through the same lens,
distortion and sensor transforms. Unresolved predictions and candidates with
inconsistent radii are rejected, rather than accepting a point source when the
Moon is absent. Shape- and brightness-valid candidates are preferred by distance
from the prediction. Similarly plausible nearby candidates cause a logged skip
rather than a guess. Unguided utility calls retain integrated-brightness ranking.

The chosen region gets proportional padding and even sensor-coordinate origins
to preserve Bayer phase. Only this crop is debayered. Its mask is refined at
native resolution with the same automatic shape checks, then clipping checks
and feathering are applied. `DEBUG` logging records the selected sensor ROI.
Guided detection permits resolved crescents with a circular outer limb and a
matching predicted radius; it does not require a full illuminated disk.
Mask dilation scales at 2% of detected radius, clamped to 1-3 pixels. Gaussian
feather radius scales at 4%, clamped to 1-4 pixels (3x3 to 9x9 kernels).
Edge padding accounts for both. Interior detail is preserved, while the limb
transition stays bounded rather than softening with an arbitrary fixed kernel.
The enable switch remains opt-in; there are no location, size, threshold, or
opacity controls.

Exposure remains manual: there is no automatic exposure adjustment. Clipped
short frames warn to reduce `EXPOSURE`; insufficient contrast warns to check
alignment before increasing it. Unresolved disks are skipped with a resolution
warning, not "recovered" by increasing their display brightness. These diagnostics
cannot prove whether clouds, focus or a bad prediction caused a failed detection.

### First-image star alignment

With Moon recovery enabled, the first eligible nighttime image is saved normally
and supplied to the existing local catalogue/star solver if there is no current
`VIRTUALSKY.ALIGNMENT` record. This is a long sky exposure, not the short Moon
exposure: stars must be visible. The first unaligned image is not composited.
Successful full solutions are saved to the configuration database together with
camera identity, reference image/sensor dimensions, binning, image-pipeline
signature, location, geometry and fit quality. Camera metadata is updated in the
same database transaction so VirtualSky can use the result. Following frames use
the saved alignment, including after a restart.

This is an explicit new alignment workflow: custom geometry values alone are
initial guesses, not proof of a completed alignment. A changed camera, image
pipeline, saved geometry or location invalidates the recorded solution and
triggers another solve. The automatic save checks for concurrent geometry edits
and merges into the latest configuration instead of replacing unrelated settings.

A cloudy, starless or only partially solved image cannot establish alignment.
The normal image remains available, the failure is logged, and automatic
alignment retries no sooner than five minutes later. Persistence failures are
also reported rather than applying an unsaved solution. There is no guarantee
the very first image contains enough stars to solve.

The initial solve is synchronous and can delay processing/cadence. It is a
one-time nighttime cost, not a per-frame solve, and may run before Moon Mode
starts. It is disabled during daytime, focus mode, incompatible 16-bit night
CLAHE, excluded diagnostic frames, or when Moon recovery is disabled.

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
later request. Exposure metadata must be finite, positive and match the request.
FITS `EXPTIME` is authoritative; `EXPOSURE` (seconds) is accepted only when
`EXPTIME` is absent. Numeric strings and driver rounding are supported, but invalid
`EXPTIME` is never hidden by an alternate field. Capture and processing share
the same matching rules. A reported duration that differs from the request
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
A timed-out bracket, or missing both exposure keywords needed to verify the pair, suspends
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

Ephemeris-guided localization predicts the sensor neighbourhood before reading
short-frame pixels. FITS `hdu.section` reads only that neighbourhood; brightness
scaling and seed-local analysis run on those pixels, not the full short image.
Only the final detected crop is debayered and blended, not a second full image
pipeline. FITS reads use `memmap=False` because Astropy rejects explicit memory
mapping for unsigned 16-bit FITS images with `BZERO` scaling. The file closes
before short-frame cleanup.
Blending reuses two ROI-sized float buffers for weighted pixels, addition and
rounding, preserving the original float32 arithmetic and round-to-even output.
Clipping rejection and alpha feathering are retained.
Timing is logged as
`Moon composite bracket=<id> applied=... in ...s`.
Disabled mode performs no extra capture, FITS read, or pixel processing.
The sky transfer is combined into one 8-bit display lookup table per frame;
the ROI needs one lookup, not another full-frame stretch or adaptive-statistics
calculation. Tables are prepared for replay only when compositing and Moon Mode
stretching are active **and that image has a paired short frame**, not for ordinary
fallback captures.

A local Windows check of the automatic localizer (20 measured iterations after
two warm-ups, NumPy 2.4.2 / OpenCV 4.13.0) used synthetic 8-bit Bayer brightness
frames with a 25-pixel-radius disk. Median / p95 times were 4.41 / 8.87 ms at
1024x1024, 14.89 / 18.53 ms at 3008x3008, and 19.58 / 24.17 ms at 4096x4096.
These measure the standalone full-frame localizer only: they exclude ephemeris
projection, first-image alignment, FITS I/O, bit-depth conversion, ROI
debayering/blending, camera acquisition and transfer. Guided operation searches
a smaller neighbourhood. These numbers are not
Raspberry Pi or real-camera performance evidence.

Historical **manual-ROI implementation** benchmarks below do not measure the
new full-frame localization cost and must not be used as its performance claim.
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
light with a circular outline can still resemble the moon: brightness and shape
are not astronomical identification. Inspect real results before enabling
unattended. Clouds, narrow crescents, unresolved disks, or multiple similar
circular lights can cause detection to be skipped, preserving the normal image.

Validate with real camera frames before relying on it. Synthetic tests exercise
pairing, skip paths, cleanup, monochrome/RGB FITS and 8/12/16-bit scaling, but do
not substitute for camera-driver and night-sky testing.
Automatic-detection tests also cover changing position and disk size, dim and
partial disks, bright craters, hot-pixel fields, non-circular distractions,
ambiguous disks, image edges, bounded previews, exact max-pool coordinates,
automatic decoder crops, and legacy configuration migration.
Guidance tests also run an actual star-catalogue solve against a synthetic sky,
check persisted alignment reuse, failed/partial solves and retry backoff,
invert image transformations and binning, verify exposure-midpoint timestamps,
ignore bright circular streetlights away from the prediction, and assert that
guided FITS reads load only the predicted section for mono/RGB/Bayer formats.
Database-save tests use test doubles; no real INDI camera is certified by these tests.

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
Capture-to-output cases send synthetic lunar/streetlight pixels through the real
capture requests and callbacks, pair their accepted FITS files, decode the normal
image, run actual ephemeris-guided localization/compositing, and save/reload a PNG.
They verify short acceptance before the long request, matching bracket IDs,
sensor-coordinate crops, one expansion for an offset Moon, rejection of remote
Moons and point sources, untouched sky and original FITS pixels, and both exposure
keywords with rounded numeric strings. No FITS pixels are rewritten after capture.
Invalid/nonfinite/nonpositive metadata for either bracket phase is also exercised
through timeout, file cleanup, reconnection and two consecutive ordinary captures.
These cases do not run the full ImageWorker rendering/output orchestration.
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
   sky/short exposures, automatically detected ROI from debug logs, and exposure period. Confirm the lunar
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
