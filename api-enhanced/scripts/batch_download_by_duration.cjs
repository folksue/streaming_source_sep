#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const { pipeline } = require('stream/promises')

const api = require('../main.js')
const generateConfig = require('../generateConfig.js')

const CONFIG = {
  // Main knobs: edit these defaults, or override them with CLI flags.
  TARGET_HOURS: 1000,
  OUTPUT_DIR: path.resolve(__dirname, '../downloads'),
  STATE_FILE: path.resolve(__dirname, '../downloads/download-state.json'),
  LEVEL: 'exhigh',
  POOL_SIZE: 100,
  PICK_SIZE: 10,

  // Operational limits.
  PLAYLIST_PAGE_LIMIT: 50,
  TRACKS_PER_PLAYLIST: 200,
  URL_BATCH_SIZE: 50,
  DOWNLOAD_CONCURRENCY: 3,
  STYLE_CONCURRENCY: 4,
  MAX_EMPTY_STYLE_ROUNDS: 8,
  API_RETRIES: 3,
  API_RETRY_DELAY_MS: 1500,

  // Leave empty to use all NetEase playlist tags under the "风格" category.
  STYLE_NAMES: [],
}

const FALLBACK_STYLES = [
  '流行',
  '摇滚',
  '民谣',
  '电子',
  '舞曲',
  '说唱',
  '轻音乐',
  '爵士',
  '乡村',
  'R&B/Soul',
  '古典',
  '民族',
  '英伦',
  '金属',
  '朋克',
  '蓝调',
  '雷鬼',
  '世界音乐',
  '拉丁',
  '另类/独立',
  'New Age',
  '古风',
  '后摇',
  'Bossa Nova',
]

function parseArgs(argv) {
  const args = {}
  for (let i = 0; i < argv.length; i += 1) {
    const item = argv[i]
    if (!item.startsWith('--')) continue

    const [rawKey, rawValue] = item.slice(2).split('=')
    const key = rawKey.replace(/-([a-z])/g, (_, char) => char.toUpperCase())
    const value = rawValue ?? argv[i + 1]
    if (rawValue === undefined) i += 1
    args[key] = value
  }
  return args
}

function applyArgs(config, args) {
  const numberMap = {
    targetHours: 'TARGET_HOURS',
    poolSize: 'POOL_SIZE',
    pickSize: 'PICK_SIZE',
    playlistPageLimit: 'PLAYLIST_PAGE_LIMIT',
    tracksPerPlaylist: 'TRACKS_PER_PLAYLIST',
    urlBatchSize: 'URL_BATCH_SIZE',
    concurrency: 'DOWNLOAD_CONCURRENCY',
    styleConcurrency: 'STYLE_CONCURRENCY',
    maxEmptyStyleRounds: 'MAX_EMPTY_STYLE_ROUNDS',
    apiRetries: 'API_RETRIES',
    apiRetryDelayMs: 'API_RETRY_DELAY_MS',
  }

  Object.entries(numberMap).forEach(([argName, configName]) => {
    if (args[argName] !== undefined) {
      const value = Number(args[argName])
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error(`Invalid --${argName}: ${args[argName]}`)
      }
      config[configName] = value
    }
  })

  if (args.outputDir) config.OUTPUT_DIR = path.resolve(args.outputDir)
  if (args.stateFile) config.STATE_FILE = path.resolve(args.stateFile)
  if (args.level) config.LEVEL = args.level
  if (args.styles) {
    config.STYLE_NAMES = args.styles
      .split(',')
      .map((style) => style.trim())
      .filter(Boolean)
  }

  return config
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true })
}

function loadState(file) {
  if (!fs.existsSync(file)) {
    return {
      downloadedIds: [],
      failedIds: {},
      totalDurationMs: 0,
      files: [],
      styleOffsets: {},
    }
  }

  const state = JSON.parse(fs.readFileSync(file, 'utf8'))
  return {
    downloadedIds: state.downloadedIds || [],
    failedIds: state.failedIds || {},
    totalDurationMs: state.totalDurationMs || 0,
    files: state.files || [],
    styleOffsets: state.styleOffsets || {},
  }
}

function saveState(file, state) {
  const tmpFile = `${file}.${process.pid}.${Date.now()}.${Math.random()
    .toString(16)
    .slice(2)}.tmp`
  fs.writeFileSync(tmpFile, `${JSON.stringify(state, null, 2)}\n`)
  fs.renameSync(tmpFile, file)
}

function shuffle(items) {
  const arr = items.slice()
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[arr[i], arr[j]] = [arr[j], arr[i]]
  }
  return arr
}

function chunk(items, size) {
  const chunks = []
  for (let i = 0; i < items.length; i += size) {
    chunks.push(items.slice(i, i + size))
  }
  return chunks
}

function sanitizeFilename(value) {
  return String(value)
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, '_')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, 160)
}

function formatDuration(ms) {
  const hours = ms / 3600000
  return `${hours.toFixed(2)}h`
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

async function withRetries(label, config, fn) {
  let lastError
  for (let attempt = 1; attempt <= config.API_RETRIES; attempt += 1) {
    try {
      return await fn()
    } catch (error) {
      lastError = error
      if (attempt < config.API_RETRIES) {
        console.warn(
          `[retry] ${label} failed (${attempt}/${config.API_RETRIES}): ${error.message}`,
        )
        await sleep(config.API_RETRY_DELAY_MS * attempt)
      }
    }
  }
  throw lastError
}

function pickExt(urlInfo) {
  if (urlInfo.type) return String(urlInfo.type).toLowerCase()
  try {
    const pathname = new URL(urlInfo.url).pathname
    const ext = path.extname(pathname).replace('.', '')
    return ext || 'mp3'
  } catch {
    return 'mp3'
  }
}

async function getStyles(config) {
  if (config.STYLE_NAMES.length > 0) return config.STYLE_NAMES

  try {
    const res = await withRetries('playlist_catlist', config, () =>
      api.playlist_catlist({ cookie: process.env.NCM_COOKIE }),
    )
    const body = res.body || {}
    const categoryMap = body.categories || {}
    const styleCategoryId = Object.entries(categoryMap).find(
      ([, name]) => name === '风格',
    )?.[0]

    const styles = (body.sub || [])
      .filter((item) => String(item.category) === String(styleCategoryId))
      .map((item) => item.name)
      .filter(Boolean)

    return styles.length > 0 ? styles : FALLBACK_STYLES
  } catch (error) {
    console.warn(`Failed to fetch playlist categories, using fallback styles: ${error.message}`)
    return FALLBACK_STYLES
  }
}

async function collectPoolForStyle(style, state, config) {
  const downloaded = new Set(state.downloadedIds)
  const failed = new Set(Object.keys(state.failedIds))
  const candidates = new Map()
  let offset = state.styleOffsets[style] || randomOffset(config.PLAYLIST_PAGE_LIMIT)
  let attempts = 0

  while (candidates.size < config.POOL_SIZE && attempts < 8) {
    let playlistRes
    try {
      playlistRes = await withRetries(`top_playlist:${style}`, config, () =>
        api.top_playlist({
          cat: style,
          order: Math.random() < 0.75 ? 'hot' : 'new',
          limit: config.PLAYLIST_PAGE_LIMIT,
          offset,
          cookie: process.env.NCM_COOKIE,
        }),
      )
    } catch (error) {
      console.warn(`Skipping playlist page for ${style}: ${error.message}`)
      offset += config.PLAYLIST_PAGE_LIMIT
      attempts += 1
      continue
    }

    const playlists = playlistRes.body?.playlists || []
    if (playlists.length === 0) {
      offset = 0
      attempts += 1
      continue
    }

    for (const playlist of shuffle(playlists)) {
      if (candidates.size >= config.POOL_SIZE) break
      try {
        const trackRes = await withRetries(`playlist_track_all:${playlist.id}`, config, () =>
          api.playlist_track_all({
            id: playlist.id,
            limit: config.TRACKS_PER_PLAYLIST,
            offset: randomOffset(Math.max(1, playlist.trackCount || 1)),
            cookie: process.env.NCM_COOKIE,
          }),
        )
        const songs = trackRes.body?.songs || []
        for (const song of shuffle(songs)) {
          if (candidates.size >= config.POOL_SIZE) break
          const id = String(song.id)
          if (downloaded.has(id) || failed.has(id) || candidates.has(id)) continue
          candidates.set(id, {
            id,
            name: song.name || id,
            artists: (song.ar || song.artists || []).map((item) => item.name).filter(Boolean),
            durationMs: Number(song.dt || song.duration || 0),
            style,
          })
        }
      } catch (error) {
        console.warn(`Skipping playlist ${playlist.id} for ${style}: ${error.message}`)
      }
    }

    offset += config.PLAYLIST_PAGE_LIMIT
    attempts += 1
  }

  state.styleOffsets[style] = offset
  return Array.from(candidates.values())
}

function getStyleDurations(state) {
  const durations = new Map()
  for (const file of state.files) {
    const current = durations.get(file.style) || 0
    durations.set(file.style, current + (Number(file.durationMs) || 0))
  }
  return durations
}

function randomOffset(max) {
  const numericMax = Number(max)
  if (!Number.isFinite(numericMax) || numericMax <= 0) return 0
  return Math.floor(Math.random() * numericMax)
}

async function getUrlMap(tracks, config) {
  const map = new Map()
  for (const group of chunk(tracks, config.URL_BATCH_SIZE)) {
    const res = await withRetries('song_url_v1', config, () =>
      api.song_url_v1({
        id: group.map((track) => track.id).join(','),
        level: config.LEVEL,
        cookie: process.env.NCM_COOKIE,
      }),
    )
    for (const item of res.body?.data || []) {
      if (item?.id && item.url) map.set(String(item.id), item)
    }
  }
  return map
}

async function downloadTrack(track, urlInfo, config, state) {
  const artist = track.artists.join(', ') || 'Unknown Artist'
  const ext = pickExt(urlInfo)
  const fileName = sanitizeFilename(`${track.id} - ${artist} - ${track.name}.${ext}`)
  const styleDir = path.join(config.OUTPUT_DIR, sanitizeFilename(track.style))
  const filePath = path.join(styleDir, fileName)

  ensureDir(styleDir)
  if (fs.existsSync(filePath)) {
    return { filePath, alreadyExists: true }
  }

  const response = await fetch(urlInfo.url, {
    headers: {
      Referer: 'https://music.163.com/',
      'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125 Safari/537.36',
    },
  })
  if (!response.ok || !response.body) {
    throw new Error(`HTTP ${response.status}`)
  }

  const tmpFile = `${filePath}.part`
  await pipeline(response.body, fs.createWriteStream(tmpFile))
  fs.renameSync(tmpFile, filePath)
  return { filePath, alreadyExists: false }
}

async function runLimited(items, concurrency, worker) {
  let nextIndex = 0
  const workers = Array.from({ length: concurrency }, async () => {
    while (nextIndex < items.length) {
      const current = items[nextIndex]
      nextIndex += 1
      await worker(current)
    }
  })
  await Promise.all(workers)
}

async function processStyle(style, context) {
  const {
    config,
    state,
    targetMs,
    perStyleTargetMs,
    reservedIds,
    styleDurations,
  } = context
  let emptyRounds = 0

  while (
    state.totalDurationMs < targetMs &&
    (styleDurations.get(style) || 0) < perStyleTargetMs &&
    emptyRounds < config.MAX_EMPTY_STYLE_ROUNDS
  ) {
    console.log(`\n[${style}] collecting pool...`)
    const pool = (await collectPoolForStyle(style, state, config)).filter(
      (track) => !reservedIds.has(track.id),
    )
    const picked = shuffle(pool).slice(0, config.PICK_SIZE)
    if (picked.length === 0) {
      emptyRounds += 1
      console.log(`[${style}] no usable tracks`)
      continue
    }

    let urlMap
    try {
      urlMap = await getUrlMap(picked, config)
    } catch (error) {
      emptyRounds += 1
      console.warn(`[${style}] failed to get song URLs: ${error.message}`)
      continue
    }
    const ready = picked.filter((track) => urlMap.has(track.id))
    for (const track of ready) reservedIds.add(track.id)

    console.log(
      `[${style}] target=${formatDuration(perStyleTargetMs)}, current=${formatDuration(
        styleDurations.get(style) || 0,
      )}, pool=${pool.length}, picked=${picked.length}, downloadable=${ready.length}`,
    )

    if (ready.length === 0) {
      emptyRounds += 1
      continue
    }

    let downloadedThisBatch = 0
    await runLimited(ready, config.DOWNLOAD_CONCURRENCY, async (track) => {
      if (
        state.totalDurationMs >= targetMs ||
        (styleDurations.get(style) || 0) >= perStyleTargetMs
      ) {
        reservedIds.delete(track.id)
        return
      }

      try {
        const urlInfo = urlMap.get(track.id)
        const result = await downloadTrack(track, urlInfo, config, state)
        if (!state.downloadedIds.includes(track.id)) {
          const durationMs = track.durationMs || 0
          state.downloadedIds.push(track.id)
          state.totalDurationMs += durationMs
          styleDurations.set(style, (styleDurations.get(style) || 0) + durationMs)
          state.files.push({
            id: track.id,
            name: track.name,
            artists: track.artists,
            style: track.style,
            durationMs,
            file: result.filePath,
            downloadedAt: new Date().toISOString(),
          })
        }
        saveState(config.STATE_FILE, state)
        downloadedThisBatch += 1
        console.log(
          `[ok] ${style} ${track.id} ${track.name} (${formatDuration(
            state.totalDurationMs,
          )} / ${config.TARGET_HOURS}h)`,
        )
      } catch (error) {
        state.failedIds[track.id] = {
          style: track.style,
          name: track.name,
          error: error.message,
          failedAt: new Date().toISOString(),
        }
        saveState(config.STATE_FILE, state)
        console.warn(`[fail] ${style} ${track.id} ${track.name}: ${error.message}`)
      } finally {
        reservedIds.delete(track.id)
      }
    })

    emptyRounds = downloadedThisBatch === 0 ? emptyRounds + 1 : 0
  }
}

async function main() {
  const config = applyArgs({ ...CONFIG }, parseArgs(process.argv.slice(2)))
  await withRetries('generateConfig', config, () => generateConfig())

  const targetMs = config.TARGET_HOURS * 3600000

  ensureDir(config.OUTPUT_DIR)
  ensureDir(path.dirname(config.STATE_FILE))

  const state = loadState(config.STATE_FILE)
  const styles = shuffle(await getStyles(config))
  const perStyleTargetMs = Math.ceil(targetMs / styles.length)
  const styleDurations = getStyleDurations(state)
  const reservedIds = new Set()

  console.log(`Target: ${config.TARGET_HOURS}h`)
  console.log(`Output: ${config.OUTPUT_DIR}`)
  console.log(`State: ${config.STATE_FILE}`)
  console.log(`Styles: ${styles.join(', ')}`)
  console.log(`Per-style target: ${formatDuration(perStyleTargetMs)}`)
  console.log(`Style concurrency: ${config.STYLE_CONCURRENCY}`)
  console.log(`Already downloaded: ${state.downloadedIds.length}, ${formatDuration(state.totalDurationMs)}`)

  await runLimited(styles, config.STYLE_CONCURRENCY, (style) =>
    processStyle(style, {
      config,
      state,
      targetMs,
      perStyleTargetMs,
      reservedIds,
      styleDurations,
    }),
  )

  console.log('\nFinished')
  console.log(`Downloaded: ${state.downloadedIds.length}`)
  console.log(`Total duration: ${formatDuration(state.totalDurationMs)}`)
  console.log(`Output: ${config.OUTPUT_DIR}`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
