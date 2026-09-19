# Dashboard action-family roadmap

The dashboard action catalog is intended to become the vocabulary for a future
planner. A plan should contain stable action IDs and typed arguments, never
Python, shell, or raw BBOS writes. `RoutineSpec` already proves the basic model:
an ordered tuple of allowlisted primitive IDs with one shared cancellation path.

## Implemented locally

| Family | Primitive IDs | Robot resource | Notes |
| --- | --- | --- | --- |
| Social gesture | `wave`, `handshake`, `fist-bump`, `hug`, `dance` | arm control/torque | Existing recordings; safe entry/return runner |
| Light expression | `light-calm`, `light-ready`, `light-thinking`, `light-celebrate`, `lights-off` | LED writer | Bounded solid, pulse, and blink effects |
| Sound cue | `sound-processing`, `sound-birthday`, `sound-low-battery` | speaker writer | Bounded PCM assets |
| Music | `music-calm`, `music-celebration` | speaker writer | Original, deterministic 16 kHz PCM assets |
| Base mode | Lean / Balance toggle | base-mode writer | Upright gate; 4° hold; balance restoration and expiry fallback |
| Routine | `welcome`, `thinking`, `celebrate`, `goodbye`, `double-wave`, `calm-moment`, `dance-party` | union of step resources | Sequential, deterministic, cancellable |

All of these work through local simulation now. Gestures had already been used
through the robot safety runner; LED, sound, and multi-step routine execution
still require the live-robot port gate.

## Best next families

These are ordered by expected product value and safety/dependency cost.

| Family | Candidate stable actions | Required gate before enabling |
| --- | --- | --- |
| Observation | `observe-people`, `observe-pose`, `capture-consented-photo`, `describe-scene` | YOLO Jetson gate; explicit camera indicator/consent; no raw-frame retention by default |
| Conversation | `listen-once`, `speak-text`, `ask-confirmation`, `interrupt-speech` | Provider-neutral audio adapter; bounded text; interruption and secret-handling tests |
| Robot status | `check-battery`, `check-upright`, `check-cameras`, `check-arm-health` | Read-only BBOS snapshot schema with freshness and units |
| More expression | `nod`, `shake-head`, `shrug`, `point-left`, `point-right`, `present-object` | New robot-specific recordings; dry run and entry-distance review for each |
| Local preferences | `remember-preference`, `forget-preference`, `list-preferences` | Local schema, consent, inspection, deletion, and retention controls |
| Reminders | `set-reminder`, `cancel-reminder`, `list-reminders` | Persistent scheduler; timezone handling; local audit trail |
| Media | `play-sound`, `stop-sound`, `set-volume` | Volume bounds and speaker ownership arbitration |
| Navigation | `turn-in-place`, `move-bounded`, `go-to-named-place`, `dock` | Obstacle sensing, localization freshness, deadman, distance/time bounds, physical e-stop operator |
| Manipulation | `home-arms`, `open-gripper`, `close-gripper`, `pick-approved-object`, `place-approved-object` | Collision/force limits, possession verification, trained-policy allowlist, workspace bounds |
| Non-medical check-in | `start-breathing-routine`, `offer-water-reminder`, `call-caregiver-with-consent` | Clear non-medical language; consent; communications authorization |

Navigation and manipulation should not be added merely as buttons backed by raw
`drive.ctrl` or arm targets. They need their own bounded hardware-owning runners
and completion evidence before joining the allowlist.

## Sequence model to grow toward

The next orchestration layer should keep routines declarative:

```json
{
  "id": "morning-welcome",
  "steps": [
    {"action": "light-ready"},
    {"action": "wave"},
    {"action": "speak-text", "args": {"text_id": "good-morning"}},
    {"action": "ask-confirmation", "args": {"prompt_id": "daily-plan"}}
  ]
}
```

Before accepting user-authored or model-authored sequences, add:

- per-action argument schemas and server-side validation;
- resource declarations (`left_arm`, `right_arm`, `speaker`, `led`, `base`,
  `camera`) and conflict checks;
- per-step timeout and a whole-routine deadline;
- confirmation levels, with human confirmation represented by an expiring
  server-issued token rather than a client boolean;
- cancellation propagation and safe cleanup for every executor;
- structured step results so conditions use facts, not log-string parsing;
- an audit record of proposed, approved, started, stopped, and completed steps;
- no arbitrary loops initially; bounded repeat counts only after time budgets
  are enforced.

Parallel steps should wait until resource declarations and cancellation are
proven. Sequential composition already covers the welcome/thinking/celebration
use cases without introducing writer conflicts.
