# Operator session checklist

**Before the participant arrives**
- [ ] Room private; door sign up; changing area ready and **out of camera view**
- [ ] Lighting set for d1 (L1); L2 variant for today noted: dim / backlit / warm (rotate)
- [ ] Two sarees chosen from the pool (A lighter or plainer, B darker or patterned); IDs noted
- [ ] Encrypted drive mounted; free space ≥ 60 GB
- [ ] Camera check: `python training/data/record_session.py --list-cameras`, then
      `--preview`: head **and feet** visible in both cameras at the floor mark
- [ ] Wi-Fi off on the recording laptop (nothing is uploaded; this makes it certain)

**With the participant**
- [ ] Info sheet given; questions answered; consent signed; consent ID written
- [ ] Row added to `trackers/sessions.csv`; participant row in `trackers/participants.csv`
- [ ] Reminder: "Tell me any time you want to pause or stop."
- [ ] Participant changes off-camera, says ready

**Recording** (`python training/data/record_session.py --out <encrypted drive> --cams 0 1 --participant P0NN --consent C-YYYY-NNN --saree-a S.. --saree-b S.. --l2 dim|backlit|warm`)
- [ ] `space` starts and stops each segment (pose clip, then d1–d4); `p` pauses for a break
- [ ] Pose clip: the participant follows the on-screen prompt; press `n` for the next prompt
- [ ] d1 correct drape: press `0`–`6` as each stage begins
- [ ] d2 mistake drape: when the prompt appears and the participant is holding the mistake,
      press `m`; press `m` again once it is corrected (~20 s later). Press `n` to skip a
      mistake that wasn't performed
- [ ] d3 correct, d4 mistakes (as above). Change lighting as shown on screen
- [ ] After each segment, read the save report printed in the terminal: `ok: true`, dropped frames
      < 2% per camera, nearest-frame offset p95 < 50 ms, and `mistakes_not_recorded` empty
      (otherwise note why)

**After**
- [ ] Compensation paid; withdrawal procedure explained
- [ ] Session notes (anything unusual: a mistake not performed, occlusions, interruptions)
- [ ] Drive unmounted and locked away; laptop copy deleted after verifying the drive copy
