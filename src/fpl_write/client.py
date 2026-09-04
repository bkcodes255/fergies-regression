"""Login test for FPL's UNOFFICIAL, undocumented account endpoints — an authenticated session
against the fantasy.premierleague.com API, not the public read-only API src/ingestion talks to.
There is no official support for this; FPL could change it or flag automated activity without
warning.

The programmatic email/password login flow (POST to users.premierleague.com/accounts/login/,
documented by older community projects like the `fpl` PyPI package) is CONFIRMED DEAD as of
2026-08-26: users.premierleague.com no longer resolves at all (checked via DNS directly, not
just blocked/CAPTCHA'd - the host is simply gone). login()/FPL_EMAIL/FPL_PASSWORD are kept
below in case that flow ever comes back, but the working path is manual cookie handoff:
login_via_cookie() uses FPL_SESSION_COOKIE (a browser session cookie you extract by hand - see
.env.example) directly as the Cookie header. This expires periodically and needs re-copying by
hand; there's no way around that without a working programmatic login.

Deliberately read-only for now: this module only verifies that authentication works
(GET /api/me/, which just confirms who you're logged in as). No transfer/lineup/captain
write call is implemented yet — see the project plan: submit-side endpoints only get built
once login itself is confirmed reliable.

Run directly (needs FPL_SESSION_COOKIE in .env - see .env.example):
    python -m src.fpl_write.client
"""
from __future__ import annotations

import requests

from config import settings

LOGIN_URL = "https://users.premierleague.com/accounts/login/"
ME_URL = "https://fantasy.premierleague.com/api/me/"
MY_TEAM_URL = "https://fantasy.premierleague.com/api/my-team/{entry_id}/"
TRANSFERS_URL = "https://fantasy.premierleague.com/api/transfers/"
USER_AGENT = "Mozilla/5.0 (fergies-regression fpl_write login test; personal account automation)"
TIMEOUT_SECONDS = 15


class FPLLoginError(RuntimeError):
    """Raised on any login failure. Never includes the password in its message."""


def login() -> requests.Session:
    """Authenticates against FPL's account login endpoint and returns a session carrying the
    resulting auth cookies. Raises FPLLoginError on any failure - wrong credentials, a
    CAPTCHA/2FA challenge, or a network error - without ever including the password in an
    exception message, log line, or print statement."""
    if not settings.FPL_EMAIL or not settings.FPL_PASSWORD:
        raise FPLLoginError(
            "FPL_EMAIL and/or FPL_PASSWORD not set in .env - see .env.example. "
            "Add your real FPL login there directly; don't paste credentials into chat."
        )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    payload = {
        "login": settings.FPL_EMAIL,
        "password": settings.FPL_PASSWORD,
        "app": "plfpl-web",
        "redirect_uri": "https://fantasy.premierleague.com/a/login",
    }
    try:
        response = session.post(LOGIN_URL, data=payload, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise FPLLoginError(f"Login request failed: {exc}") from exc

    # A 200 here does NOT mean login succeeded - FPL's login page returns 200 with an
    # error embedded in the HTML body (wrong password, CAPTCHA challenge) rather than a
    # non-2xx status. The only reliable success signal is the auth cookie actually being set.
    if "pl_profile" not in session.cookies.get_dict() and "sessionid" not in session.cookies.get_dict():
        body_lower = response.text.lower()
        if "captcha" in body_lower or "recaptcha" in body_lower:
            raise FPLLoginError(
                "Login blocked by a CAPTCHA challenge - this account/IP needs a manual "
                "browser login first, or automated login isn't viable for this account."
            )
        raise FPLLoginError(
            "Login did not set an auth cookie - likely incorrect email/password. "
            f"HTTP status was {response.status_code}."
        )

    return session


def login_via_cookie() -> requests.Session:
    """Builds a session authenticated via manually-extracted browser credentials instead of the
    dead POST-login flow. The Cookie header alone is NOT sufficient (confirmed empirically
    2026-08-26 - a cookie-only session got a 200 with player=null) - FPL's rebuilt frontend
    authenticates API calls via a custom `X-Api-Authorization` header, not cookies. Sends both:
    the cookie (harmless, may still be checked for CSRF/bot-protection reasons) plus the real
    auth header. Both are set as raw header values rather than parsed, since a copy-pasted
    browser value may not survive being re-split and reassembled identically."""
    if not settings.FPL_API_AUTHORIZATION:
        raise FPLLoginError(
            "FPL_API_AUTHORIZATION not set in .env - see .env.example. Extract it from a real "
            "browser session: F12 -> Network -> Fetch/XHR filter -> the request to "
            "https://fantasy.premierleague.com/api/me/ -> Request Headers -> x-api-authorization."
        )
    session = requests.Session()
    headers = {"User-Agent": USER_AGENT, "X-Api-Authorization": settings.FPL_API_AUTHORIZATION}
    if settings.FPL_SESSION_COOKIE:
        headers["Cookie"] = settings.FPL_SESSION_COOKIE
    session.headers.update(headers)
    return session


def verify_login(session: requests.Session) -> dict:
    """Confirms the session is actually authenticated by calling the read-only /api/me/
    endpoint. Returns the parsed player info dict. Raises FPLLoginError if the session
    turns out not to be authenticated despite login() appearing to succeed (e.g. a cookie
    was set but the server still rejects API calls)."""
    try:
        response = session.get(ME_URL, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise FPLLoginError(f"/api/me/ request failed: {exc}") from exc

    if response.status_code != 200:
        raise FPLLoginError(
            f"/api/me/ returned HTTP {response.status_code} - session is not authenticated "
            "despite login() setting a cookie."
        )
    data = response.json()
    if "player" not in data:
        raise FPLLoginError(f"/api/me/ returned 200 but no 'player' key - unexpected shape: {list(data.keys())}")
    return data


def get_my_team(session: requests.Session, entry_id: int) -> dict:
    """GET /api/my-team/{id}/ - your CURRENT, private, in-progress squad state (transfer bank,
    free transfers, picks with pending changes) - distinct from the public
    entry/{id}/event/{event}/picks/ endpoint src/ingestion already uses, which only ever shows
    a gameweek's already-submitted, finalized picks. This one requires real account auth (not
    just being a public API), so a successful call is a much stronger signal that write-scoped
    auth works than /api/me/ alone - but it's still a GET, so still zero risk. Raises
    FPLLoginError on any non-200."""
    url = MY_TEAM_URL.format(entry_id=entry_id)
    try:
        response = session.get(url, timeout=TIMEOUT_SECONDS)
    except requests.RequestException as exc:
        raise FPLLoginError(f"/api/my-team/ request failed: {exc}") from exc
    if response.status_code != 200:
        raise FPLLoginError(
            f"/api/my-team/{entry_id}/ returned HTTP {response.status_code} - this endpoint "
            "needs real account-scoped auth, not just login; body: " + response.text[:300]
        )
    return response.json()


def save_my_team_noop(session: requests.Session, entry_id: int, my_team: dict) -> requests.Response:
    """POSTs the exact current picks (element/position/multiplier/is_captain/is_vice_captain)
    back to /api/my-team/{id}/ completely unchanged - a genuine no-op write test, not a guess
    at a real transfer/captain change. Unverified endpoint/shape (no captured real example was
    available - see project memory) - this IS the test of whether that guess is right. Returns
    the raw response rather than raising, so the caller can inspect a failure's exact body."""
    picks_payload = [
        {
            "element": p["element"],
            "position": p["position"],
            "multiplier": p["multiplier"],
            "is_captain": p["is_captain"],
            "is_vice_captain": p["is_vice_captain"],
        }
        for p in my_team["picks"]
    ]
    url = MY_TEAM_URL.format(entry_id=entry_id)
    return session.post(url, json={"picks": picks_payload}, timeout=TIMEOUT_SECONDS)


def set_captain(
    session: requests.Session, entry_id: int, my_team: dict, captain_element: int, vice_captain_element: int
) -> requests.Response:
    """POSTs the same picks as save_my_team_noop, except is_captain/is_vice_captain are set on
    captain_element/vice_captain_element instead of wherever they currently are - a real
    change, not a no-op. Both elements must already be among the 15 picks (this only reassigns
    the captain/VC flags within the existing squad, it doesn't add/remove players - that's a
    transfer, a different, not-yet-built operation)."""
    picks_payload = [
        {
            "element": p["element"],
            "position": p["position"],
            "multiplier": p["multiplier"],
            "is_captain": p["element"] == captain_element,
            "is_vice_captain": p["element"] == vice_captain_element,
        }
        for p in my_team["picks"]
    ]
    url = MY_TEAM_URL.format(entry_id=entry_id)
    return session.post(url, json={"picks": picks_payload}, timeout=TIMEOUT_SECONDS)


def build_transfer_payload(
    entry_id: int, event_id: int, out_element: int, in_element: int,
    purchase_price: int, selling_price: int,
) -> dict:
    """Builds the /api/transfers/ payload without sending it - the only genuinely SAFE way to
    inspect what a transfer call would send. Do NOT treat "confirmed": False as a safe preview:
    confirmed empirically on 2026-08-26 (scripts/test_transfer_dryrun.py, scripts/
    revert_transfer.py) that FPL executes the transfer for real regardless of the confirmed
    flag's value. That mistake cost a real, non-refundable transfer plus a locked-in -4 hit for
    the gameweek, with no API-level undo - recorded as a hard rule since: never POST to this
    endpoint outside a real, deliberate, user-approved submission. This function exists so a
    caller can inspect/log the exact payload before that POST happens, without touching the
    network at all."""
    return {
        "confirmed": True,
        "entry": entry_id,
        "event": event_id,
        "transfers": [
            {
                "element_in": in_element,
                "element_out": out_element,
                "purchase_price": purchase_price,
                "selling_price": selling_price,
            }
        ],
        "wildcard": False,
        "freehit": False,
    }


def submit_transfer(
    session: requests.Session, entry_id: int, event_id: int,
    out_element: int, in_element: int, purchase_price: int, selling_price: int,
) -> requests.Response:
    """POSTs a single transfer to /api/transfers/ - REAL and IRREVERSIBLE the instant this is
    called (see build_transfer_payload's docstring - there is no safe dry-run for this call
    itself). The endpoint/payload shape is field-verified, not guessed: this exact shape is what
    scripts/test_transfer_dryrun.py's accidental real transfer and scripts/revert_transfer.py's
    revert both went through with. Not verified: every edge case (price changes mid-window,
    wildcard/free-hit interaction, multi-transfer arrays). purchase_price/selling_price are
    FPL's tenths-of-a-million integers (e.g. 125 = £12.5m) - pass whatever a fresh get_my_team()
    call reports for the outgoing pick's selling price and the incoming player's current cost,
    never a value read off potentially-stale ingested DB prices. Returns the raw response rather
    than raising, so the caller can inspect a failure's exact body."""
    payload = build_transfer_payload(entry_id, event_id, out_element, in_element, purchase_price, selling_price)
    return session.post(TRANSFERS_URL, json=payload, timeout=TIMEOUT_SECONDS)


def run() -> None:
    if settings.FPL_API_AUTHORIZATION:
        print("Using FPL_API_AUTHORIZATION (manual credential handoff)...")
        session = login_via_cookie()
    else:
        print("No FPL_API_AUTHORIZATION set - falling back to the (likely dead) POST login flow...")
        session = login()
    print("Verifying via GET /api/me/ (read-only)...")
    me = verify_login(session)
    player = me["player"]
    print(
        f"Login verified. Authenticated as: {player.get('first_name')} {player.get('last_name')} "
        f"(entry: {me.get('player', {}).get('entry')})"
    )
    entry_id = me["player"]["entry"]
    print(f"Fetching private /api/my-team/{entry_id}/ (still read-only, but account-scoped auth)...")
    my_team = get_my_team(session, entry_id)
    picks = my_team.get("picks", [])
    transfers = my_team.get("transfers", {})
    print(
        f"my-team fetched OK: {len(picks)} picks, "
        f"bank={transfers.get('bank')}, free transfers={transfers.get('limit')}, "
        f"transfers made this GW={transfers.get('made')}"
    )
    print(
        "No write action was attempted - both calls above are GETs. Submit-side endpoints "
        "(transfers/lineup/captain POST) are not implemented yet."
    )


if __name__ == "__main__":
    run()
