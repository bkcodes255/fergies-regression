// Run after a build: fails if any public file in dist/ holds the owner's private data. Everything
// counts as public except dist/this-week/ and dist/season/, including the shared /_astro/ bundles.
//
// A report names the signal's source, never its text: CI logs on this public repo are public too.
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { dirname, extname, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

// JSON keys that only private exporters write. Player names are public, so they are not signals.
const PRIVATE_KEYS = [
  "board_calls", "after_board", "free_transfers", "bank_before", "bank_after", "runs_if_silent",
  "hit_cost", "total_hit_cost", "total_net_gain", "uses_free_transfer", "xi_points_before", "xi_points_after",
];
const PRIVATE_FILES = ["this-week.json", "sample-this-week.json", "season.json", "sample-season.json"];
const PRIVATE_DIRS = new Set(["this-week", "season"]);
const FREE_TEXT_KEYS = new Set(["title", "detail", "shout", "mutter"]);
const BINARY = new Set([".woff", ".woff2", ".png", ".jpg", ".jpeg", ".webp", ".avif", ".gif", ".ico"]);
// Shorter strings, or single words, could turn up on a public page by chance.
const isFreeText = (s) => typeof s === "string" && s.trim().length >= 12 && /\s/.test(s.trim());

const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: '"', apos: "'" };
const JS_ESCAPES = { n: " ", r: " ", t: " " };

/** Undoes HTML and JSON escaping and evens out case and spacing, so an escaped copy still matches. */
export function normalise(text) {
  return text
    .replace(/&(?:(amp|lt|gt|quot|apos)|#(\d+)|#x([0-9a-f]+));/gi, (_, name, dec, hex) =>
      name ? ENTITIES[name.toLowerCase()] : String.fromCodePoint(dec ? Number(dec) : parseInt(hex, 16)))
    .replace(/\\(?:u([0-9a-f]{4})|(.))/gis, (_, code, ch) =>
      code ? String.fromCharCode(parseInt(code, 16)) : JS_ESCAPES[ch] ?? ch)
    .replace(/\s+/g, " ")
    .toLowerCase();
}

function freeText(node, where, boardCall, out) {
  if (Array.isArray(node)) {
    node.forEach((v, i) => freeText(v, `${where}[${i}]`, boardCall, out));
  } else if (node && typeof node === "object") {
    for (const [k, v] of Object.entries(node)) {
      const path = `${where}.${k}`;
      if ((FREE_TEXT_KEYS.has(k) || (boardCall && k === "label")) && isFreeText(v)) out.push({ text: v, where: path });
      else freeText(v, path, boardCall || k === "board_calls", out);
    }
  }
}

/** Every signal to look for: { needle, name }. Throws if there is nothing to compare against. */
export function loadSignals(dataDir) {
  const found = PRIVATE_FILES.filter((f) => existsSync(join(dataDir, f)));
  if (!found.length) throw new Error(`No private data files in ${dataDir}, so there is nothing to check against.`);
  const signals = new Map(PRIVATE_KEYS.map((k) => [k, `private key "${k}"`]));
  for (const file of found) {
    const texts = [];
    freeText(JSON.parse(readFileSync(join(dataDir, file), "utf8")), file, false, texts);
    for (const { text, where } of texts) {
      const needle = normalise(text);
      if (!signals.has(needle)) signals.set(needle, `free text at ${where}`);
    }
  }
  return [...signals].map(([needle, name]) => ({ needle, name }));
}

function publicFiles(root, dir = root) {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) return dir === root && PRIVATE_DIRS.has(entry.name) ? [] : publicFiles(root, path);
    return BINARY.has(extname(entry.name).toLowerCase()) ? [] : [path];
  });
}

/** @returns {{ scanned: number, leaks: { file: string, signal: string }[] }} */
export function findLeaks(distDir, dataDir) {
  if (!existsSync(distDir)) throw new Error(`${distDir} doesn't exist: run the build first.`);
  const signals = loadSignals(dataDir);
  const files = publicFiles(distDir);
  if (!files.some((f) => f.endsWith(".html"))) throw new Error(`No public HTML in ${distDir}: the build looks wrong.`);
  const leaks = [];
  for (const file of files) {
    const haystack = normalise(readFileSync(file, "utf8"));
    for (const { needle, name } of signals) {
      if (haystack.includes(needle)) leaks.push({ file: relative(distDir, file).split(sep).join("/"), signal: name });
    }
  }
  return { scanned: files.length, leaks };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const web = join(dirname(fileURLToPath(import.meta.url)), "..");
  try {
    const { scanned, leaks } = findLeaks(join(web, "dist"), join(web, "src", "data"));
    for (const { file, signal } of leaks) console.error(`LEAK dist/${file}: ${signal}`);
    if (leaks.length) {
      console.error(`${leaks.length} private value(s) found in public files. Do not deploy this build.`);
      process.exit(1);
    }
    console.log(`No private data in ${scanned} public files.`);
  } catch (err) {
    console.error(`Leak check could not run: ${err.message}`);
    process.exit(2);
  }
}
