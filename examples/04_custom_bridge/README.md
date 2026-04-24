# Writing a custom bridge

Bridges are just classes implementing `bridge(BridgeInput) -> BridgeOutput`.
This example registers a tiny one that adds a colour-temperature word, then uses
it through the registry by name.

```bash
python run.py
```
