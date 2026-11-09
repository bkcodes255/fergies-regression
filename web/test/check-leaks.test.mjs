import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { findLeaks } from "../scripts/check-leaks.mjs";

const THIS_WEEK = {
  squad: {
    board_calls: [{ label: "Rogers, Reed L on, King, Adams O off" }],
    calls: [
      { title: "Sell Reed L, buy Lewis", detail: "It's the model's top pick & more", shout: "Why is Reed L still in my squad? Sell him.", mutter: "Good. One more thing: Lewis." },
      { title: "Armband", detail: "short" },
    ],
  },
};

/** Builds a fake site and data folder, runs the check, and cleans up. */
function scan(files, data = { "sample-this-week.json": THIS_WEEK }) {
  const root = mkdtempSync(join(tmpdir(), "leaks-"));
  try {
    const dist = join(root, "dist");
    const dataDir = join(root, "data");
    mkdirSync(dataDir, { recursive: true });
    for (const [name, value] of Object.entries(data)) writeFileSync(join(dataDir, name), JSON.stringify(value));
    for (const [name, text] of Object.entries({ "index.html": "<p>Salah and Haaland</p>", ...files })) {
      mkdirSync(dirname(join(dist, name)), { recursive: true });
      writeFileSync(join(dist, name), text);
    }
    return findLeaks(dist, dataDir);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

test("a clean site passes", () => assert.deepEqual(scan({ "fixtures/index.html": "<p>Reed L v Lewis</p>" }).leaks, []));
test("private pages are not scanned", () => {
  const { leaks } = scan({ "this-week/index.html": "board_calls Sell Reed L, buy Lewis", "season/index.html": "after_board" });
  assert.deepEqual(leaks, []);
});
test("finds a private key name in a public page or bundle", () => {
  assert.equal(scan({ "fixtures/index.html": '{"after_board":1}' }).leaks.length, 1);
  assert.equal(scan({ "_astro/page.abc.js": "x.free_transfers" }).leaks[0].file, "_astro/page.abc.js");
});
test("finds free text however it was escaped", () => {
  for (const text of [
    "Sell Reed L, buy Lewis",
    "sell   reed l,\n buy lewis",
    "It&#39;s the model&#39;s top pick &amp; more",
    'It\\u0027s the model\\u0027s top pick \\u0026 more',
    "Why is Reed L still in my squad? Sell him.",
    "Good. One more thing: Lewis.",
    "Rogers, Reed L on, King, Adams O off",
  ]) {
    assert.equal(scan({ "model-lab/index.html": `<p>${text}</p>` }).leaks.length, 1, text);
  }
});
test("short or single-word text is not a signal, and a report hides the text", () => {
  assert.deepEqual(scan({ "fixtures/index.html": "<p>Armband short</p>" }).leaks, []);
  const [leak] = scan({ "fixtures/index.html": "Sell Reed L, buy Lewis" }).leaks;
  assert.equal(leak.signal, "free text at sample-this-week.json.squad.calls[0].title");
});
test("a directory merely named like a private page is still scanned", () => {
  assert.equal(scan({ "fixtures/this-week/index.html": "hit_cost" }).leaks.length, 1);
  assert.equal(scan({ "this-week.html": "hit_cost" }).leaks.length, 1);
});
test("reads whichever private files exist", () => {
  const season = { calls: [{ title: "Wildcard the whole team now" }] };
  const hit = (data) => scan({ "fixtures/index.html": "Wildcard the whole team now" }, data).leaks.length;
  assert.equal(hit({ "sample-this-week.json": THIS_WEEK }), 0);
  assert.equal(hit({ "sample-this-week.json": THIS_WEEK, "sample-season.json": season }), 1);
  assert.equal(hit({ "season.json": season }), 1);
});
test("fails closed with no private data or no build", () => {
  assert.throws(() => scan({}, {}), /nothing to check against/);
  assert.throws(() => findLeaks(join(tmpdir(), "no-such-dist-here"), tmpdir()), /run the build first/);
});
