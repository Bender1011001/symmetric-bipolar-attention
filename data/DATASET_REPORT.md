# ECoT Dataset Report

Seed: `20260526`
Total records: `55000`
Written duplicate text hashes: `0`
Skipped duplicate candidates: `0`

## Splits

- `test`: `2500`
- `train`: `50000`
- `validation`: `2500`

## Pillars

- `backtracking` (Algorithmic Backtracking): `13750`
- `belief_revision` (Non-Monotonic Belief Revision): `13750`
- `mutual_exclusivity` (Strict Mutual Exclusivity): `13750`
- `system1_honeypot` (System 1 Honeypot): `13750`

## Templates

- `backtracking_dependency`: `2750`
- `backtracking_graph_path`: `2750`
- `backtracking_latin_square`: `2750`
- `backtracking_maze`: `2750`
- `backtracking_sum_assignment`: `2750`
- `belief_access_control`: `2750`
- `belief_game_rule`: `2750`
- `belief_refund`: `2750`
- `belief_safety_protocol`: `2750`
- `belief_shipping`: `2750`
- `mutual_crypto_digits`: `2750`
- `mutual_inventory_route`: `2750`
- `mutual_machines`: `2750`
- `mutual_rooms`: `2750`
- `mutual_schedule`: `2750`
- `system_average_speed`: `2750`
- `system_bat_ball`: `2750`
- `system_bayes_base_rate`: `2750`
- `system_code_precedence`: `2750`
- `system_percent_reversal`: `2750`

## Validation Contract

Every generated record is passed through `validate_tagged_text` before it is written.
The validator enforces exactly one required block, required reasoning tag order, non-empty content, and explicit eliminative language inside `<INHIBIT>`.

## Quality Design

- System 1 examples use exact arithmetic, Bayes conditioning, operator precedence, and equal-distance speed traps.
- Mutual exclusivity examples use one-to-one assignment constraints across rooms, machines, schedules, code digits, and route resources.
- Belief revision examples activate a general rule, then apply a later narrower exception.
- Backtracking examples explicitly follow a tempting branch into a concrete contradiction, then reset to a valid branch or report inconsistency.
