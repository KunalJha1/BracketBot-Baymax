# Judge demo orchestration

The dashboard's **Judge demo** is a small state machine, not a timed macro. It
keeps one clear story running while still allowing unscripted questions:

## First-minute opening

Choose **Start first minute + robot demo** to start the shared presenter clock
and the safe robot sequence. The dashboard advances an operator-only cue card;
it does not synthesize the presenter's lines or fake a reminder.

- **0:00 — You to BracketBot:** “BracketBot, remind me in 3 minutes to take my
  medication.” Pause for the real spoken confirmation.
- **0:12 — Presenter to audience:** “Hi everyone! Meet BracketBot. BracketBot
  is an at-home care assistant that can answer questions, converse with you,
  and even do tasks for you.”
- **0:35 — Presenter to audience:** “While that reminder runs, let's see what
  else BracketBot can do.” Continue into questions or safe light and sound
  features while the manipulation task runs.

The three-minute reminder deliberately outlives the opening so it can land later
as proof that BracketBot keeps a background care task running during other work.
Three minutes places delivery in the middle of the packing phase, while the arms
are visibly busy and nobody has touched the dashboard — the strongest moment for
it to interrupt. Nothing in the dashboard creates or fakes the reminder: the
presenter speaks the line, the robot parses it locally, and the SQLite scheduler
owns the countdown from there. When it lands, BracketBot plays a short chime,
blinks amber for eight seconds, and speaks “Reminder: take my medication.”
Use the reminder only as a demo coordination aid, not as a medical schedule.

The physical sequence remains event-driven:

1. **Meet Baymax** — ready light, then a wave.
2. **Pack while we talk** — the packing policy owns the arms and cameras while
   conversation, lights, sounds, and music remain available.
3. **Finish together** — only after the packing process has released its robot
   resources, Baymax runs the celebration routine.

The operator may choose **Confirm box is packed** when the physical result is
visible. Confirmation sends `SIGINT` to the configured packing process and
waits for it to exit; it does not skip cleanup or jump straight to the finale.
The packing adapter must therefore treat `SIGINT` as “home/release safely, then
exit.” A zero exit status is the completion evidence accepted by the
orchestrator. The global Stop uses the same cancellation path without running
the finale.

## The network-free path

Every spoken beat that the presenter controls should be a pre-rendered line
from the **Lines** family, not a live model answer. **Introduce Baymax** (`H`)
plays the self-introduction and waves; **What I can do** (`A`) gives the
capability tour; **Sign off** (`J`) closes. These need no network, no wake
word, and no transcription, and they stay available while the packing policy
owns the arms.

Keep the live conversation for the one unscripted judge question. Warm
`assets/response-cache-seed.json` (loaded automatically at greeter startup) so
the common questions are answered from the cache, and rehearse one spoken
question beforehand so Whisper and the TTS cache are warm. If the link is
down, say so plainly and use a line; do not retry a dead model call on stage.

## Rehearse without the robot

```sh
python3 scripts/robot_dashboard.py --simulate
```

Open <http://127.0.0.1:8020/>, choose **Start judge demo**, try light and sound
actions during the packing phase, verify that arm actions are disabled, then
choose **Confirm box is packed**.

## Connect the real packing policy

Start the dashboard with one trusted command whose lifetime represents one
packing attempt:

```sh
python3 scripts/robot_dashboard.py \
  --demo-pack-command '/absolute/path/to/the-safe-pack-wrapper'
```

The command is parsed into an argument vector and launched without a shell. A
wrapper is appropriate when the policy needs environment setup, a tunnel, or a
remote robot process. Its contract is:

- start one bounded packing attempt;
- stream useful progress to stdout;
- exit `0` only after success and robot-resource release;
- exit nonzero on failure;
- on `SIGINT`, home/release the arms before exiting;
- never return while a child process still owns an arm writer.

Do not point the dashboard at a launcher that merely backgrounds the real
policy and exits. The orchestrator would mistake that launcher exit for task
completion.

During packing, the dashboard rejects every action that declares an arm or
camera channel. For the entire composed sequence it also holds
`/tmp/bracketbot-demo-arm-reserved` on the robot. The local and Gemini voice
gesture paths check that reservation, so a spoken movement request is refused
while ordinary Q&A remains available.

## Two-minute presenter shape

- “This is Baymax, our embodied assistant.” Start the demo.
- As packing begins: “The manipulation policy has the arms; the assistant is
  still responsive.” Ask a natural project question.
- Trigger one non-motion expression, such as the thinking light or processing
  sound. Avoid stacking every feature into the same run.
- Explain the safety boundary: the model selects named tools, while deterministic
  runners own hardware and Stop propagates to the active task.
- When the item is visibly in the box, confirm completion. Let the automatic
  celebration land, then stop talking. The packed box is the final proof.

Keep a no-network answer and a manual Stop rehearsal ready. If packing fails,
say what safety gate rejected it; do not hide the failure by forcing the finale.
