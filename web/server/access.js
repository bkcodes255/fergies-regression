// The login check behind every request: Cloudflare Access signs the visitor in (one-time PIN to
// an allowed email) and forwards a signed JWT in the Cf-Access-Jwt-Assertion header. This
// verifies that token - signature against the team's published keys, issuer, audience, expiry -
// so the site fails closed: a hostname Access isn't protecting (the *.pages.dev address, a
// preview deployment) or a missing setting serves nothing, rather than the squad to anyone.
import { createRemoteJWKSet, jwtVerify } from "jose";

const keySets = new Map();

/** Settings from the Pages project's environment variables. */
export function accessConfig(env) {
  const team = String(env.CF_ACCESS_TEAM_DOMAIN || "").trim().replace(/^https?:\/\//, "").replace(/\/+$/, "");
  const audiences = String(env.CF_ACCESS_AUD || "").split(",").map((s) => s.trim()).filter(Boolean);
  const emails = String(env.ALLOWED_EMAILS || "").split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
  return { issuer: team ? `https://${team}` : "", audiences, emails };
}

function teamKeys(issuer) {
  if (!keySets.has(issuer)) keySets.set(issuer, createRemoteJWKSet(new URL(`${issuer}/cdn-cgi/access/certs`)));
  return keySets.get(issuer);
}

/**
 * @param {Request} request
 * @param {Record<string, string>} env
 * @param {{ key?: any }} [opts] key: verification key override, for tests only.
 * @returns {Promise<{ ok: true, email: string } | { ok: false, status: number, reason: string }>}
 */
export async function verifyAccess(request, env, opts = {}) {
  const { issuer, audiences, emails } = accessConfig(env);
  if (!issuer || !audiences.length) {
    return { ok: false, status: 500, reason: "Login isn't configured: set CF_ACCESS_TEAM_DOMAIN and CF_ACCESS_AUD." };
  }
  const token = request.headers.get("Cf-Access-Jwt-Assertion");
  if (!token) return { ok: false, status: 403, reason: "Sign in required." };
  let payload;
  try {
    ({ payload } = await jwtVerify(token, opts.key ?? teamKeys(issuer), {
      issuer, audience: audiences, algorithms: ["RS256"], clockTolerance: 30,
    }));
  } catch {
    return { ok: false, status: 403, reason: "Sign in required." };
  }
  const email = String(payload.email || "").toLowerCase();
  // Access's own policy decides who may sign in; this is a second lock in case that policy is
  // ever loosened by mistake.
  if (emails.length && !emails.includes(email)) return { ok: false, status: 403, reason: "This account isn't allowed." };
  return { ok: true, email };
}
