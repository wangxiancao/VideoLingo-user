# Triage Labels

The skills speak in terms of five canonical triage roles. This file maps those roles to the GitHub labels used in this repo.

| Label in mattpocock/skills | Label in our tracker | Meaning                                  |
| -------------------------- | -------------------- | ---------------------------------------- |
| `needs-triage`             | `needs-triage`       | Maintainer needs to evaluate this issue  |
| `needs-info`               | `needs-info`         | Waiting on reporter for more information |
| `ready-for-agent`          | `ready-for-agent`    | Fully specified, ready for an AFK agent  |
| `ready-for-human`          | `ready-for-human`    | Requires human implementation            |
| `wontfix`                  | `wontfix`            | Will not be actioned                     |

When a skill mentions a role (e.g. "apply the AFK-ready triage label"), use the corresponding GitHub label:

```bash
gh issue edit <number> --add-label "ready-for-agent"
```

The four labels other than `wontfix` were created on 2026-10-21; `wontfix` already existed. Edit the table if the vocabulary changes.