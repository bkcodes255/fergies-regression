// Runs in front of every request to the site (Cloudflare Pages Functions), static files included.
import { verifyAccess } from "../server/access.js";

const HEADERS = {
  "X-Robots-Tag": "noindex, nofollow",
  "Referrer-Policy": "no-referrer",
  "X-Content-Type-Options": "nosniff",
  "Content-Security-Policy": "frame-ancestors 'none'",
};

export async function onRequest({ request, env, next }) {
  const check = await verifyAccess(request, env);
  if (!check.ok) {
    return new Response(check.reason, {
      status: check.status, headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store", ...HEADERS },
    });
  }
  const res = await next();
  const out = new Response(res.body, res);
  for (const [k, v] of Object.entries(HEADERS)) out.headers.set(k, v);
  // Never let a shared cache keep a copy of a signed-in page.
  out.headers.set("Cache-Control", "private, no-store");
  return out;
}
