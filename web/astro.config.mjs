// @ts-check
import { defineConfig } from "astro/config";

// A fully static site: the data is baked in at build time (see src/lib/data.ts), and the only
// code that runs on Cloudflare is the login check in functions/_middleware.js.
export default defineConfig({
  output: "static",
  build: { inlineStylesheets: "auto" },
  devToolbar: { enabled: false },
});
