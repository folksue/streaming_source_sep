# Batch Downloader

This repository now includes a standalone script for downloading music by total duration.

Script:

`scripts/batch_download_by_duration.cjs`

Run it with:

```bash
npm run download:duration
```

## What it does

- Estimates a per-style target from the requested total duration.
- Samples playlists and tracks in randomized pools.
- Downloads tracks concurrently.
- Skips duplicate track IDs using a state file.
- Saves tracks into per-style folders under the output directory.
- Rescans the output directory on startup so you can stop and restart without losing progress.

## Main settings

Edit these defaults near the top of `scripts/batch_download_by_duration.cjs`:

- `TARGET_HOURS`: total duration target in hours, default `1000`
- `OUTPUT_DIR`: where downloaded files are written
- `STATE_FILE`: JSON state used to skip duplicates and resume progress
- `LEVEL`: quality level, default `exhigh`
- `POOL_SIZE`: candidate pool size per style
- `PICK_SIZE`: how many tracks to sample from each pool
- `DOWNLOAD_CONCURRENCY`: concurrent file downloads
- `STYLE_CONCURRENCY`: concurrent styles processed at once
- `API_RETRIES`: retry count for NetEase API calls
- `RESCAN_ON_START`: rescan the output directory and rebuild progress from existing files

You can also override the most important ones from the CLI:

```bash
npm run download:duration -- \
  --target-hours 1000 \
  --level lossless \
  --pool-size 100 \
  --pick-size 10 \
  --style-concurrency 4
```

## Cookie

The script reads `NCM_COOKIE` from `.env`.

Use the minimal cookie fields needed for login and high-quality requests, typically:

- `MUSIC_U`
- `__csrf`

`.env` is ignored by git and should stay local.

## File types

- `lossless` and higher levels can produce real `.flac` files for some tracks.
- Some tracks still resolve to `.mp3`, depending on availability and rights.
- The script does not force-convert formats. It saves whatever the source URL returns.
- On restart, it counts the existing files under each style folder and resumes from that total.

## Notes

- The `downloads-test/` directory is only for local verification.
- Do not commit downloaded media files.
- The API service itself is unchanged; this adds a helper script and a `package.json` entry only.
