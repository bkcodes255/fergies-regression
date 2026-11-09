import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { exportJWK, generateKeyPair, SignJWT } from "jose";
import { isPublicRequest, isPublicUrl } from "../server/routes.js";
import { onRequest } from "../functions/_middleware.js";

const ORIGIN = "https://fergies.briankileo.com";
const TEAM = "fergie.cloudflareaccess.com";
const env = { CF_ACCESS_TEAM_DOMAIN: TEAM, CF_ACCESS_AUD: "aud-main" };
const { publicKey, privateKey } = await generateKeyPair("RS256");

const PUBLIC_PATHS = [
  "/", "/index.html",
  "/fixtures", "/fixtures/", "/fixtures/index.html",
  "/optimal-squad", "/optimal-squad/", "/optimal-squad/index.html",
  "/model-lab", "/model-lab/", "/model-lab/index.html",
  "/favicon.svg",
  "/_astro/Base.Bx_1-2.css", "/_astro/anton-latin-400-normal.a1B2c3.woff2", "/_astro/page.js", "/_astro/nested/chunk.a1B2.js",
  "/?utm_source=a%20b", "/fixtures/?gw=8", "/_astro/page.js?v=1",
];

// Every one of these reaches the middleware as written: Request would tidy some of them first.
const PRIVATE_PATHS = [
  // Login pages and paths nobody listed
  "/this-week", "/this-week/", "/this-week/index.html", "/season", "/season/", "/season/index.html",
  "/nope", "/robots.txt", "/404.html", "/favicon.ico", "/favicon.svg/", "/fixtures.html", "/fixtures/extra", "/fixtures/extra.html",
  "/_astro", "/_astro/", "/_astro/page", "/_astro/nested/",
  // Case changes
  "/FIXTURES/", "/Fixtures/", "/fixtures/INDEX.HTML", "/INDEX.HTML", "/FAVICON.SVG", "/_ASTRO/page.js", "/_Astro/page.js",
  "/THIS-WEEK/", "/This-Week/",
  // Duplicate slashes
  "//", "//fixtures/", "/fixtures//", "//favicon.svg", "/fixtures//index.html", "//_astro/page.js", "/_astro//page.js", "/_astro/nested//page.js",
  "//this-week/",
  // Encoded slashes
  "/fixtures%2F", "/fixtures%2f", "%2Ffixtures/", "/_astro%2Fpage.js", "/_astro/nested%2fpage.js", "/_astro/x%2F..%2F..%2Fthis-week%2Findex.html",
  "/_astro/..%2fthis-week/index.html", "/_astro/..%2Fthis-week/index.html", "/_astro/page.js%2F..%2F..%2Fseason%2F",
  // Backslashes, raw and encoded
  "/fixtures\\", "\\fixtures/", "/_astro\\page.js", "/_astro/..\\this-week\\index.html", "/_astro/page.js\\..\\..\\this-week\\", "/fixtures/?a\\b",
  "/_astro/..%5cthis-week/index.html", "/_astro/..%5Cthis-week%5Cindex.html", "/fixtures%5C",
  // Dots, raw and encoded
  "/_astro/../this-week/index.html", "/_astro/x/../../this-week/index.html", "/_astro/./page.js", "/_astro/..", "/_astro/.", "/_astro/.hidden", "/_astro/.hidden.js",
  "/_astro/page..js", "/_astro/page.js.", "/_astro/a..b/page.js",
  "/fixtures/.", "/fixtures/..", "/fixtures/./", "/fixtures/../this-week/", "/fixtures./", "/./fixtures/", "/../fixtures/", "/this-week/../fixtures/",
  "/_astro/%2e%2e/this-week/index.html", "/_astro/%2E%2E/this-week/index.html", "/_astro/.%2e/this-week/index.html", "/_astro/%2e./this-week/index.html",
  "/_astro/%2e/page.js", "/fixtures/%2e%2e/this-week/", "/fixtures%2e/",
  // Double encoding, and encoding plain letters
  "/_astro/%252e%252e/this-week/index.html", "/_astro/..%252fthis-week/index.html", "/%66ixtures/", "/fixtures/index%2ehtml", "/%5Fastro/page.js", "/_astro/page%2ejs",
  "/%2e", "/%2F", "/index%2Ehtml",
  // Control characters, spaces, unicode, and other separators
  "/fix\ttures/", "/fixtures/\n", "\t/fixtures/", "/fixtures/ ", "/fixtures /", "/fixtures%20/", "/fixtures%00/", "/_astro/page.js%00.css", "/_astro/page.js\0",
  "/\uff46ixtures/", "/ﬁxtures/", "/fixtures\u2215", "/_astro/page\u2024js",
  "/fixtures;/", "/fixtures/;x", "/_astro/page.js;x", "/_astro/page.js;/../../this-week/", "/_astro/~page.js", "/_astro/@page.js", "/_astro/page:x.js",
  // Things that look like a query or fragment hiding a path
  "/fixtures/%3f/../../this-week/", "/fixtures/%23/../../this-week/", "/fixtures/#/../this-week/",
];

// A plain object, so the URL arrives exactly as written. Request's constructor would normalise it.
const raw = (path, { method = "GET", headers = {} } = {}) => ({ url: `${ORIGIN}${path}`, method, headers: new Headers(headers) });
const token = (claims = {}) =>
  new SignJWT({ email: "gaffer@example.com", ...claims }).setProtectedHeader({ alg: "RS256" })
    .setIssuer(`https://${TEAM}`).setAudience("aud-main").setIssuedAt().setExpirationTime("5m").sign(privateKey);

function run(request, e = env, res = () => new Response("page body", { status: 200, headers: { "Content-Type": "text/html" } })) {
  const calls = { next: 0 };
  return onRequest({ request, env: e, next: async () => { calls.next += 1; return res(); } }).then((out) => ({ out, calls }));
}

function assertSecurityHeaders(res, label) {
  assert.equal(res.headers.get("X-Robots-Tag"), "noindex, nofollow", label);
  assert.equal(res.headers.get("Referrer-Policy"), "no-referrer", label);
  assert.equal(res.headers.get("X-Content-Type-Options"), "nosniff", label);
  assert.equal(res.headers.get("Content-Security-Policy"), "frame-ancestors 'none'", label);
}

async function assertRefused(request, status, label) {
  const { out, calls } = await run(request);
  assert.equal(out.status, status, label);
  assert.equal(calls.next, 0, `${label}: the page must not be fetched`);
  assert.equal(out.headers.get("Cache-Control"), "no-store", label);
  assertSecurityHeaders(out, label);
}

// verifyAccess fetches the team's keys, so answer that one URL and refuse any other.
const realFetch = globalThis.fetch;
before(async () => {
  const jwk = await exportJWK(publicKey);
  globalThis.fetch = async (url) => {
    assert.equal(String(url), `https://${TEAM}/cdn-cgi/access/certs`);
    return new Response(JSON.stringify({ keys: [jwk] }), { status: 200, headers: { "Content-Type": "application/json" } });
  };
});
after(() => { globalThis.fetch = realFetch; });

test("every public form is public", () => {
  for (const p of PUBLIC_PATHS) assert.equal(isPublicUrl(`${ORIGIN}${p}`), true, p);
});
test("every private or odd form is private", () => {
  for (const p of PRIVATE_PATHS) assert.equal(isPublicUrl(`${ORIGIN}${p}`), false, JSON.stringify(p));
});
test("malformed URLs are private", () => {
  for (const u of [undefined, null, 7, "", "/fixtures/", "fixtures/", "ftp://x/fixtures/", "HTTPS://x/fixtures/", "https://", "https:///fixtures/", `${ORIGIN}`, `${ORIGIN}?a=1`, `${ORIGIN}\\fixtures/`]) {
    assert.equal(isPublicUrl(u), false, String(u));
  }
});
test("only GET and HEAD are public", () => {
  assert.equal(isPublicRequest(raw("/fixtures/")), true);
  assert.equal(isPublicRequest(raw("/fixtures/", { method: "HEAD" })), true);
  for (const method of ["POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "get", ""]) {
    assert.equal(isPublicRequest(raw("/fixtures/", { method })), false, method);
  }
});

test("middleware: public forms open with no login settings, cached briefly", async () => {
  for (const p of PUBLIC_PATHS) {
    for (const e of [{}, env]) {
      const { out, calls } = await run(raw(p), e);
      assert.equal(out.status, 200, p);
      assert.equal(calls.next, 1, p);
      assert.equal(await out.text(), "page body", p);
      assert.equal(out.headers.get("Cache-Control"), "public, max-age=300", p);
      assertSecurityHeaders(out, p);
    }
  }
});
test("middleware: HEAD works on a public page", async () => {
  const { out } = await run(raw("/fixtures/", { method: "HEAD" }), {});
  assert.equal(out.status, 200);
});
test("middleware: a public page ignores any token sent with it", async () => {
  const { out, calls } = await run(raw("/fixtures/", { headers: { "Cf-Access-Jwt-Assertion": "garbage" } }), {});
  assert.equal(out.status, 200);
  assert.equal(calls.next, 1);
});
test("middleware: only a 200 or 304 from a public path is cacheable", async () => {
  const cases = [[304, "public, max-age=300"], [301, "no-store"], [308, "no-store"], [404, "no-store"], [500, "no-store"]];
  for (const [status, cache] of cases) {
    const body = status === 304 ? null : "x";
    const { out } = await run(raw("/fixtures/"), {}, () => new Response(body, { status }));
    assert.equal(out.status, status);
    assert.equal(out.headers.get("Cache-Control"), cache, String(status));
    assertSecurityHeaders(out, String(status));
  }
});
test("middleware: public cache headers replace whatever the asset sent", async () => {
  const { out } = await run(raw("/fixtures/"), {}, () => new Response("x", { headers: { "Cache-Control": "public, max-age=31536000", "X-Robots-Tag": "all" } }));
  assert.equal(out.headers.get("Cache-Control"), "public, max-age=300");
  assert.equal(out.headers.get("X-Robots-Tag"), "noindex, nofollow");
});

test("middleware: every private or odd form is refused without a token", async () => {
  for (const p of PRIVATE_PATHS) await assertRefused(raw(p), 403, JSON.stringify(p));
});
test("middleware: paths a real Request tidies up stay private", async () => {
  for (const p of ["/_astro/../this-week/", "/fixtures/../this-week/index.html", "/_astro/%2e%2e/this-week/", "/_astro/.%2e/season/", "/_astro\\..\\this-week\\", "/FIXTURES/", "//fixtures/"]) {
    await assertRefused(new Request(`${ORIGIN}${p}`), 403, p);
  }
});
test("middleware: writes to public pages need a login", async () => {
  for (const method of ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"]) {
    await assertRefused(raw("/fixtures/", { method }), 403, method);
    await assertRefused(new Request(`${ORIGIN}/`, { method }), 403, method);
  }
});
test("middleware: private pages fail closed when login isn't configured", async () => {
  for (const e of [{}, { CF_ACCESS_TEAM_DOMAIN: TEAM }, { CF_ACCESS_AUD: "aud-main" }]) {
    for (const p of ["/this-week/", "/season/", "/nope", "/FIXTURES/"]) {
      const { out, calls } = await run(raw(p, { headers: { "Cf-Access-Jwt-Assertion": await token() } }), e);
      assert.equal(out.status, 500, p);
      assert.equal(calls.next, 0, p);
      assert.equal(out.headers.get("Cache-Control"), "no-store", p);
      assertSecurityHeaders(out, p);
    }
  }
});
test("middleware: a bad token gets nothing", async () => {
  const other = await generateKeyPair("RS256");
  const forged = await new SignJWT({ email: "x@example.com" }).setProtectedHeader({ alg: "RS256" })
    .setIssuer(`https://${TEAM}`).setAudience("aud-main").setIssuedAt().setExpirationTime("5m").sign(other.privateKey);
  for (const jwt of ["garbage", forged]) {
    await assertRefused(raw("/this-week/", { headers: { "Cf-Access-Jwt-Assertion": jwt } }), 403, jwt.slice(0, 10));
  }
});
test("middleware: a valid token opens a private page, never cached", async () => {
  for (const p of ["/this-week/", "/season/", "/season/index.html", "/nope"]) {
    const { out, calls } = await run(raw(p, { headers: { "Cf-Access-Jwt-Assertion": await token() } }));
    assert.equal(out.status, 200, p);
    assert.equal(calls.next, 1, p);
    assert.equal(await out.text(), "page body", p);
    assert.equal(out.headers.get("Cache-Control"), "private, no-store", p);
    assertSecurityHeaders(out, p);
  }
});
test("middleware: ALLOWED_EMAILS still applies to private pages", async () => {
  const locked = { ...env, ALLOWED_EMAILS: "gaffer@example.com" };
  const ok = await run(raw("/this-week/", { headers: { "Cf-Access-Jwt-Assertion": await token() } }), locked);
  assert.equal(ok.out.status, 200);
  const no = await run(raw("/this-week/", { headers: { "Cf-Access-Jwt-Assertion": await token({ email: "someone@example.com" }) } }), locked);
  assert.equal(no.out.status, 403);
  assert.equal(no.calls.next, 0);
});
