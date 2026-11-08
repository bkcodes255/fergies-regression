// Runs in front of every request to the site (Cloudflare Pages Functions), static files included.
// Public files pass straight through; everything else needs a valid Cloudflare Access login.
import { verifyAccess } from "../server/access.js";
import { isPublicRequest } from "../server/routes.js";

const HEADERS = {
  "X-Robots-Tag": "noindex, nofollow",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "Content-Security-Policy": "frame-ancestors 'none'",
};

function withHeaders(res, cacheControl) {
  const out = new Response(res.body, res);
  for (const [k, v] of Object.entries(HEADERS)) out.headers.set(k, v);
  out.headers.set("Cache-Control", cacheControl);
  return out;
}

export async function onRequest({ request, env, next }) {
  if (isPublicRequest(request)) {
    const res = await next();
    // Only a successful answer may be kept, and only briefly. 304 refreshes a copy already held.
    return withHeaders(res, res.status === 200 || res.status === 304 ? "public, max-age=300" : "no-store");
  }
  const check = await verifyAccess(request, env);
  if (!check.ok) {
    return new Response(check.reason, {
      status: check.status, headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store", ...HEADERS },
    });
  }
  // Never let a shared cache keep a copy of a signed-in page.
  return withHeaders(await next(), "private, no-store");
}
