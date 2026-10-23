# Historical code provenance

The frozen physical action library and the current development scripts are not interchangeable.

| Item | SHA-256 | Release interpretation |
|---|---|---|
| Historical annual-library script recorded by the frozen run package | `c1b0c7a94cfd906568aa9f9f3367120badd299ac2bc7daa73e58a4469ab96c63` | Provenance record only; the exact generator snapshot is unavailable. |
| Later working-tree script with the same filename | `433e5cc2702df509a6e93d74acbe18abd19d0a76de82506295cf3ee259387070` | Must not be represented as reconstructing the frozen action library. |

Consequently, Level 1 starts from frozen hashed analysis products, while optional Level 2 reruns frozen IDF/EPW trajectories. Neither level retunes building physics or generates new actions. This limitation is disclosed consistently in the README and future output manifest.
