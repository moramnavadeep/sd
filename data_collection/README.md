# data_collection/

Everything needed to run recording sessions. The plan is in ../docs/DATA_COLLECTION_PLAN.md.

| File | Use |
|---|---|
| protocol.yaml | drapes, lighting, cameras, deliberate-mistake scripts and their rotation (read by the recorder) |
| participant_info_sheet.md, consent_form_template.md | TEMPLATES for the ethics committee; fill the [brackets] |
| session_checklist.md | operator checklist for each session |
| trackers/*.csv | participants, sessions, saree pool, annotation time log. Copy consent_register.TEMPLATE.csv to consent_register.csv and keep the filled copy OFF this repo (it is git-ignored) |

Recordings are written by `training/data/record_session.py` to a directory you choose (put it on the
encrypted drive), never inside this repository.
