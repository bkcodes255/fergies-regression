// Which requests may skip the login. An allow-list, so a page nobody thought about needs a login.
//
// The check reads the URL exactly as it arrived rather than trusting new URL(): Cloudflare's asset
// server may decode a path differently (%2f is left alone by new URL(), but decoded it turns
// /_astro/..%2fthis-week/ into a private page). So only a path already in its one plain spelling
// passes. Anything encoded, doubled, re-cased or otherwise odd falls through to the login.

const PUBLIC_PATHS = new Set([
  "/",
  "/index.html",
  "/favicon.svg",
  ...["fixtures", "optimal-squad", "model-lab"].flatMap((p) => [`/${p}`, `/${p}/`, `/${p}/index.html`]),
]);

// Build output under /_astro/: plain names with an extension, split by single dots, so a segment
// can't be empty, be "..", or start or end with a dot.
const ASSET = /^\/_astro\/(?:[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*\/)*[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)+$/;

// Anything outside visible ASCII, or a backslash, in any part of the URL.
const ODD_CHARACTER = /[^\x21-\x7e]|\\/;

// origin, path, optional query. A fragment never reaches a server, so one doesn't match.
const URL_PARTS = /^https?:\/\/[^/?#]+(\/[^?#]*)(?:\?[^#]*)?$/;

/** True only if this exact URL string names a public file. Pure: no I/O, no settings. */
export function isPublicUrl(url) {
  if (typeof url !== "string" || ODD_CHARACTER.test(url)) return false;
  const path = URL_PARTS.exec(url)?.[1];
  if (path === undefined || !(PUBLIC_PATHS.has(path) || ASSET.test(path))) return false;
  // A public path needs no rewriting, so if the URL parser would change it, it isn't what it seems.
  try {
    return new URL(url).pathname === path;
  } catch {
    return false;
  }
}

/** True only for a GET or HEAD of a public file. Every other request needs a valid login. */
export function isPublicRequest(request) {
  return (request.method === "GET" || request.method === "HEAD") && isPublicUrl(request.url);
}
