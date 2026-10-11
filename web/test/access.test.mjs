import { test } from "node:test";
import assert from "node:assert/strict";
import { generateKeyPair, SignJWT } from "jose";
import { verifyAccess } from "../server/access.js";

const env = { CF_ACCESS_TEAM_DOMAIN: "fergie.cloudflareaccess.com", CF_ACCESS_AUD: "aud-main, aud-other" };
const { publicKey, privateKey } = await generateKeyPair("RS256");
const other = await generateKeyPair("RS256");

const token = (claims = {}, { key = privateKey, iss = "https://fergie.cloudflareaccess.com", aud = "aud-main", exp = "5m" } = {}) =>
  new SignJWT({ email: "Gaffer@Example.com", ...claims }).setProtectedHeader({ alg: "RS256" })
    .setIssuer(iss).setAudience(aud).setIssuedAt().setExpirationTime(exp).sign(key);
const req = (jwt) => new Request("https://fergie.pages.dev/", { headers: jwt ? { "Cf-Access-Jwt-Assertion": jwt } : {} });
const check = async (jwt, e = env) => verifyAccess(req(jwt), e, { key: publicKey });

test("fails closed when login isn't configured", async () => {
  assert.equal((await check(await token(), {})).status, 500);
  assert.equal((await check(await token(), { CF_ACCESS_TEAM_DOMAIN: "x.cloudflareaccess.com" })).status, 500);
});
test("refuses a request with no Access token", async () => assert.equal((await check(null)).status, 403));
test("accepts a valid token, for any configured audience", async () => {
  assert.deepEqual(await check(await token()), { ok: true, email: "gaffer@example.com" });
  assert.equal((await check(await token({}, { aud: "aud-other" }))).ok, true);
});
test("accepts the team domain written as a URL", async () => {
  assert.equal((await check(await token(), { ...env, CF_ACCESS_TEAM_DOMAIN: "https://fergie.cloudflareaccess.com/" })).ok, true);
});
test("refuses the wrong audience, issuer, signer, or an expired token", async () => {
  assert.equal((await check(await token({}, { aud: "someone-else" }))).status, 403);
  assert.equal((await check(await token({}, { iss: "https://evil.cloudflareaccess.com" }))).status, 403);
  assert.equal((await check(await token({}, { key: other.privateKey }))).status, 403);
  assert.equal((await check(await token({}, { exp: Math.floor(Date.now() / 1000) - 120 }))).status, 403);
});
test("refuses a tampered token", async () => {
  const [h, p, s] = (await token()).split(".");
  const forged = Buffer.from(JSON.stringify({ ...JSON.parse(Buffer.from(p, "base64url")), email: "intruder@example.com" })).toString("base64url");
  assert.equal((await check(`${h}.${forged}.${s}`)).status, 403);
});
test("refuses an unsigned (alg: none) token", async () => {
  const b = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
  const now = Math.floor(Date.now() / 1000);
  const none = `${b({ alg: "none" })}.${b({ email: "x@example.com", iss: "https://fergie.cloudflareaccess.com", aud: "aud-main", exp: now + 300 })}.`;
  assert.equal((await check(none)).status, 403);
});
test("ALLOWED_EMAILS is a second lock, case-insensitive", async () => {
  const locked = { ...env, ALLOWED_EMAILS: "gaffer@example.com" };
  assert.equal((await check(await token(), locked)).ok, true);
  const r = await check(await token({ email: "someone@example.com" }), locked);
  assert.equal(r.status, 403);
});
