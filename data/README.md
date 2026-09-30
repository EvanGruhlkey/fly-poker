# Connectome data

The MaleCNS dataset is not committed. Build the local sparse bundle with:

```powershell
uv run python -m fly_poker.connectome.fetch
```

The fetcher must verify the source license and the checksum recorded in
`data/malecns/manifest.json` before replacing the local bundle.
