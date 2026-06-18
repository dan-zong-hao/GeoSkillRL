# SkillRL Round0 Audit

This report is generated offline. It does not mutate source RL/SFT data.

## Data Summary

- failures: 4752
- skill SFT rows: 2164
- bbox RL rows: 1870
- seed skills: 16

## Skill Coverage

- failure rows with retrieved skills: 3903 (82.13%)
- skill SFT rows with retrieved skills: 1614 (74.58%)
- bbox RL rows with retrieved skills: 1512 (80.86%)

## Failure Metrics

- bbox_wrong failures: 4150
- spatial violation failures: 2183
- deer-horse rate among failures: 2705 (56.92%)
- false-grounded correct among failures: 1493 (31.42%)

## Top Failure Types

| failure_type | count |
|---|---:|
| bbox_wrong | 4150 |
| deer_horse_crop_claim | 2705 |
| false_grounded_correct | 1493 |
| top_violated | 420 |
| bottom_violated | 304 |
| corner_partial:bottom_violated | 265 |
| corner_partial:top_violated | 253 |
| corner_partial:left_violated | 226 |
| corner_partial:right_violated | 185 |
| left_violated | 149 |
| right_violated | 94 |
| bottom_violated+left_violated | 74 |
| top_violated+left_violated | 62 |
| bottom_violated+right_violated | 59 |
| top_violated+right_violated | 49 |
| missing_gt | 12 |
| missing_bbox | 9 |
| corner_partial:bottom_violated+right_violated | 7 |
| corner_partial:bottom_violated+left_violated | 6 |
| corner_partial:top_violated+right_violated | 5 |

## Spatial Violation Distribution

| spatial_violation | count |
|---|---:|
| none | 2569 |
| top_violated | 420 |
| bottom_violated | 304 |
| corner_partial:bottom_violated | 265 |
| corner_partial:top_violated | 253 |
| corner_partial:left_violated | 226 |
| corner_partial:right_violated | 185 |
| left_violated | 149 |
| right_violated | 94 |
| bottom_violated+left_violated | 74 |
| top_violated+left_violated | 62 |
| bottom_violated+right_violated | 59 |
| top_violated+right_violated | 49 |
| missing_gt | 12 |
| missing_bbox | 9 |
| corner_partial:bottom_violated+right_violated | 7 |
| corner_partial:bottom_violated+left_violated | 6 |
| corner_partial:top_violated+right_violated | 5 |
| corner_partial:top_violated+left_violated | 4 |

## Comparative Locator Backlog

| backlog_family | count |
|---|---:|
| largest | 571 |
| smallest | 8 |

## Top Skills Retrieved

| skill_id | count |
|---|---:|
| dir_corner_instance | 1464 |
| dir_top_extremum_instance | 622 |
| rank_largest_instance | 528 |
| dir_bottom_extremum_instance | 439 |
| rel_above_below_anchor | 342 |
| subpart_side_locator | 341 |
| dir_left_extremum_instance | 277 |
| dir_right_extremum_instance | 274 |
| rel_adjacent_context | 256 |
| rel_left_of_anchor | 93 |
| rel_right_of_anchor | 89 |
| rel_front_context | 53 |

## Top Categories

| category | count |
|---|---:|
| Object relative position / context | 619 |
| Object color / pattern | 595 |
| Object shape / structure | 524 |
| Object state / motion / activity | 515 |
| Object category refinement | 430 |
| Object material / surface | 316 |
| Region function | 316 |
| Other visual features | 288 |
| Object function / usage | 286 |
| Object category | 276 |
| Object existence | 241 |
| Region status | 193 |
| Counting | 153 |
