"""Scrapes club crests (SVG + PNG) from a footylogos.com competition page and uploads them
straight into a private Supabase Storage bucket - never written to disk in the repo, never
public. Run this LOCALLY (the Claude Code sandbox that authored this script has no egress to
footylogos.com, so the site's actual markup was never inspected - verify the discovered links
with --debug on first run and adjust TEAM_LINK_PATTERN below if footylogos.com nests crests
under a path this pattern doesn't match).

Setup:
    pip install -r requirements.txt
    Add to .env (see .env.example):
        SUPABASE_URL=https://<project-ref>.supabase.co
        SUPABASE_SERVICE_ROLE_KEY=<service_role key, from Project Settings -> API - NEVER the
            anon/publishable key, and never commit this>
        SUPABASE_CRESTS_BUCKET=team-crests   # already created as a private bucket

Run from the repo root:
    python scripts/scrape_crests_to_supabase.py --debug   # inspect discovered links first
    python scripts/scrape_crests_to_supabase.py           # download + upload for real
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings

DEFAULT_URL = "https://www.footylogos.com/competition/premier-league"
USER_AGENT = "Mozilla/5.0 (compatible; fergies-regression-crest-fetch/1.0)"
CRESTS_ROOT = "crests"

# Links on the competition page that lead to a per-club page (as opposed to a direct asset
# link). Loosened on purpose - tighten it if the real page also links to unrelated /club/...
# style URLs that aren't crest pages.
TEAM_LINK_PATTERN = re.compile(r"/(club|team|logo|badge)/", re.IGNORECASE)
ASSET_SUFFIX_PATTERN = re.compile(r"\.(svg|png)(\?.*)?$", re.IGNORECASE)


@dataclass
class CrestAsset:
    team_slug: str
    url: str
    ext: str


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def fetch(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def guess_team_name(link_tag, page_url: str) -> str:
    text = link_tag.get_text(strip=True)
    if text:
        return text
    img = link_tag.find("img")
    if img and img.get("alt"):
        return img["alt"]
    return Path(urlparse(page_url).path).stem


def discover_assets(session: requests.Session, competition_url: str, debug: bool) -> list[CrestAsset]:
    soup = fetch(session, competition_url)
    assets: dict[tuple[str, str], CrestAsset] = {}
    team_pages: dict[str, str] = {}

    for a in soup.find_all("a", href=True):
        href = urljoin(competition_url, a["href"])
        if ASSET_SUFFIX_PATTERN.search(href):
            ext = ASSET_SUFFIX_PATTERN.search(href).group(1).lower()
            team = slugify(guess_team_name(a, competition_url))
            assets[(team, ext)] = CrestAsset(team, href, ext)
        elif TEAM_LINK_PATTERN.search(href):
            team = slugify(guess_team_name(a, competition_url))
            if team:
                team_pages.setdefault(team, href)

    if debug:
        print(f"[debug] direct asset links found on competition page: {len(assets)}")
        print(f"[debug] candidate team pages found: {len(team_pages)}")
        for team, url in team_pages.items():
            print(f"[debug]   {team} -> {url}")

    for team, page_url in team_pages.items():
        try:
            team_soup = fetch(session, page_url)
        except requests.RequestException as exc:
            print(f"[warn] could not fetch team page for {team} ({page_url}): {exc}")
            continue
        for a in team_soup.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            match = ASSET_SUFFIX_PATTERN.search(href)
            if match:
                ext = match.group(1).lower()
                assets.setdefault((team, ext), CrestAsset(team, href, ext))
        time.sleep(0.5)  # be polite to the site between per-team page fetches

    return list(assets.values())


def upload_to_supabase(assets: list[CrestAsset], session: requests.Session, bucket: str) -> None:
    from supabase import create_client

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_ROLE_KEY:
        raise SystemExit(
            "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set - fill them in .env first."
        )

    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)
    storage = client.storage.from_(bucket)

    content_types = {"svg": "image/svg+xml", "png": "image/png"}
    ok, failed = 0, 0
    for asset in assets:
        try:
            resp = session.get(asset.url, headers={"User-Agent": USER_AGENT}, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            print(f"[error] download failed for {asset.team_slug}.{asset.ext}: {exc}")
            failed += 1
            continue

        path = f"{CRESTS_ROOT}/{asset.team_slug}/{asset.team_slug}.{asset.ext}"
        try:
            storage.upload(
                path,
                resp.content,
                {"content-type": content_types[asset.ext], "upsert": "true"},
            )
            print(f"[ok] {path} ({len(resp.content)} bytes)")
            ok += 1
        except Exception as exc:  # supabase-py raises its own StorageException
            print(f"[error] upload failed for {path}: {exc}")
            failed += 1

    print(f"\nDone: {ok} uploaded, {failed} failed, bucket '{bucket}' (private).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Competition page to scrape")
    parser.add_argument(
        "--bucket",
        default=settings.SUPABASE_CRESTS_BUCKET,
        help="Target private Supabase Storage bucket (must already exist)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print discovered links and exit without downloading/uploading anything",
    )
    args = parser.parse_args()

    session = requests.Session()
    assets = discover_assets(session, args.url, debug=args.debug)

    if not assets:
        print("No crest links found - the site's markup likely differs from what "
              "TEAM_LINK_PATTERN / ASSET_SUFFIX_PATTERN expect. Re-run with --debug and "
              "adjust the patterns at the top of this file.")
        return

    print(f"Discovered {len(assets)} crest files across "
          f"{len({a.team_slug for a in assets})} teams.")

    if args.debug:
        for asset in assets:
            print(f"  {asset.team_slug}.{asset.ext} -> {asset.url}")
        return

    upload_to_supabase(assets, session, args.bucket)


if __name__ == "__main__":
    main()
