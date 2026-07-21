# ECoT Quality Audit

Records audited: `55000`
Invalid records: `0`
Duplicate text hashes: `0`
Character length range: `545` to `1323`
Mean character length: `971`

## Forbidden Phrase Hits

- None

## Pillar Counts

- `backtracking`: `13750`
- `belief_revision`: `13750`
- `mutual_exclusivity`: `13750`
- `system1_honeypot`: `13750`

## Template Counts

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

## Sampled Record IDs

- `backtracking_dependency`: `ecot-backtracking-000671-3218ace2fc8b8614`, `ecot-backtracking-042011-52201c46c938c8f5`, `ecot-backtracking-041851-7fd96f443f093544`
- `backtracking_graph_path`: `ecot-backtracking-051555-a35902fc5f0d07b9`, `ecot-backtracking-051235-05fe44bfc2380247`, `ecot-backtracking-002475-c7f68e08056e20d7`
- `backtracking_latin_square`: `ecot-backtracking-032839-c4667e11fd537ea1`, `ecot-backtracking-028479-21947bea263db1fb`, `ecot-backtracking-003579-34570162375942fd`
- `backtracking_maze`: `ecot-backtracking-006023-4269057a3a118b42`, `ecot-backtracking-054243-de1e48b5b4647bba`, `ecot-backtracking-045523-7cd9e620b59c29cc`
- `backtracking_sum_assignment`: `ecot-backtracking-030367-59ed96cb865d64d5`, `ecot-backtracking-047007-d85431b247558b72`, `ecot-backtracking-037287-714d537f8cb38695`
- `belief_access_control`: `ecot-belief_revision-021074-fcdb27b823ab63cb`, `ecot-belief_revision-018574-1b203764ad6a2732`, `ecot-belief_revision-031614-05e9ccdc350d8045`
- `belief_game_rule`: `ecot-belief_revision-003158-0a1ba47e71b69f5d`, `ecot-belief_revision-033378-f7046b3d5643a5b3`, `ecot-belief_revision-040958-7992c88e816a314a`
- `belief_refund`: `ecot-belief_revision-049526-741dd661cb25d982`, `ecot-belief_revision-012586-06035c71d363fcb7`, `ecot-belief_revision-054786-e1dad45014189054`
- `belief_safety_protocol`: `ecot-belief_revision-004750-e2c004eef344ec84`, `ecot-belief_revision-042550-e94a636e731e1eae`, `ecot-belief_revision-025770-5973482da4028621`
- `belief_shipping`: `ecot-belief_revision-011042-299d8320c450a062`, `ecot-belief_revision-030242-fffa9513544ff3b5`, `ecot-belief_revision-016502-69b7b0a19417aa16`
- `mutual_crypto_digits`: `ecot-mutual_exclusivity-002913-57100e2f0d712cd6`, `ecot-mutual_exclusivity-051973-4bfdd7b907dee0bc`, `ecot-mutual_exclusivity-027033-6076cee99f6dbe3b`
- `mutual_inventory_route`: `ecot-mutual_exclusivity-032837-5093b961824c8588`, `ecot-mutual_exclusivity-009337-30bc5230f735b724`, `ecot-mutual_exclusivity-034337-ac96696aecbb70a1`
- `mutual_machines`: `ecot-mutual_exclusivity-051765-c59c8cc3c9dd41b6`, `ecot-mutual_exclusivity-037905-52a38d2a9a87779d`, `ecot-mutual_exclusivity-009265-77cd97529f4da8ce`
- `mutual_rooms`: `ecot-mutual_exclusivity-014441-b96f193a8ec60ce3`, `ecot-mutual_exclusivity-047281-e2bbb9fecfea26bf`, `ecot-mutual_exclusivity-054901-969c75418d802cd1`
- `mutual_schedule`: `ecot-mutual_exclusivity-053889-dd4692dde61e4a02`, `ecot-mutual_exclusivity-048049-47df6877d10bb42c`, `ecot-mutual_exclusivity-040129-669c4e6dcd2ec916`
- `system_average_speed`: `ecot-system1_honeypot-004028-6618d9bcecee1f98`, `ecot-system1_honeypot-050288-af1ead11c6ca1c73`, `ecot-system1_honeypot-014328-0027d18752a35240`
- `system_bat_ball`: `ecot-system1_honeypot-040240-9c83d63c570bf7ca`, `ecot-system1_honeypot-047000-2c3e932703bd87a0`, `ecot-system1_honeypot-053560-69c4ce9c772c6b44`
- `system_bayes_base_rate`: `ecot-system1_honeypot-018652-cd05c7dd7f2143c1`, `ecot-system1_honeypot-030492-43e921b1c0509ec8`, `ecot-system1_honeypot-049912-2206230aa826c0ae`
- `system_code_precedence`: `ecot-system1_honeypot-038336-2bf6aee1dbb42f4b`, `ecot-system1_honeypot-020616-f51e13650383373f`, `ecot-system1_honeypot-041816-67df54889efc4b0c`
- `system_percent_reversal`: `ecot-system1_honeypot-032044-c9f6d98e337548d1`, `ecot-system1_honeypot-034904-36b0605f419bf3db`, `ecot-system1_honeypot-002884-d1e40f1967dbf21a`

## Critique

- The dataset now avoids the earlier artificial global case-prefix pattern; any identifiers are domain-specific to the problem context.
- The previous medical-medication template was removed to avoid training clinical action advice.
- The previous inconsistent Latin-square template was repaired so backtracking reaches a valid legal symbol instead of teaching impossible-puzzle answers as a routine pattern.
- Remaining examples are intentionally scaffolded and repetitive at the tag level. That is desirable for the inhibition hook, but natural-language variety should be expanded in a later frontier-authored augmentation pass.
