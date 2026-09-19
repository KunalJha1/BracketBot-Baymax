# Baymax Bot

## Inspiration

We watched *Big Hero 6* and thought Baymax was pretty tuff. We wanted to recreate that same calm, friendly energy in a real robot—one that can interact with people through conversation, expressive movements, lights, and sounds while keeping safety at the center of every action.

## What it does

Baymax Bot turns BracketBot into an interactive companion that can:

- Respond to voice commands and hold conversations
- Perform gestures such as waving, handshakes, fist bumps, hugs, and dancing
- Run multi-step routines like a welcome sequence, calm moment, and dance party
- Express itself using lights, sound effects, and original music
- Detect people and visible facial expressions using a fully local vision pipeline
- Be controlled through an accessible web dashboard
- Stop actions immediately through a global emergency stop control

Every physical action passes through a fixed allowlist and deterministic safety checks. The language model can suggest an action, but it never directly controls the robot's motors.

## How we built it

We built Baymax Bot in Python on top of BracketBot OS. A local web dashboard communicates with the robot over SSH and provides controls for gestures, lights, sounds, music, routines, and balance mode.

Gestures are based on recorded arm trajectories and executed by dedicated robot-side runners. Before moving, the system checks the robot's orientation, validates movement limits, prevents conflicting actions, and smoothly enters and exits each motion.

For voice interaction, we used Gemini Live for speech input and output, while OpenRouter handles general conversation. Spoken movement requests are matched against a strict local command list so an AI-generated response cannot accidentally trigger the robot.

Our local perception system combines YOLO for person detection, YuNet for face localization, and EmotiEffLib for visible-expression estimation. We also created a simulation mode so we could test the complete dashboard, API, sequencing, progress tracking, and cancellation flow without connecting to the physical robot.

## Challenges we ran into

The biggest challenge was making a powerful robot feel expressive without sacrificing safety. Recorded gestures begin from specific positions, so we had to account for the robot's current arm height and create smooth transitions into and out of each movement.

Finally, hardware access was limited at times. Building a realistic simulation mode allowed us to continue developing and testing without bypassing the safety checks used on the real robot.

## Accomplishments that we're proud of

We are especially proud of building a complete interaction system instead of a single scripted demo. Baymax Bot includes 15 primitive actions, seven multi-step routines, voice commands, local perception, balance control, and an accessible dashboard.

## What we learned

We learned that robotics is less about making one movement work and more about making every transition, interruption, and failure predictable.

We also learned that AI should not have unrestricted access to physical hardware. Separating conversational intelligence from deterministic robot control gave us the personality of an AI assistant without allowing generated output to become an unsafe motor command.

## What's next for Baymax Bot

Next, we want to add more natural speech interruption and turn-taking, consent-aware vision, authenticated remote access, and locally stored user preferences with clear privacy controls.

We also plan to expand Baymax Bot with safe navigation, obstacle detection, additional expressive gestures, and robot health telemetry. Eventually, we want it to combine conversation, perception, and approved actions into helpful routines while remaining transparent, interruptible, and safe around people.
