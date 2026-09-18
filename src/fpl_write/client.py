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


def set_lineup(
    session: requests.Session, entry_id: int, my_team: dict,
    starting_elements: list[int], bench_elements: list[int],
) -> requests.Response:
    """POSTs the same /api/my-team/{id}/ endpoint as save_my_team_noop/set_captain (the endpoint
    already proven safe to write to repeatedly - this is a NEW use of it, not a new endpoint),
    reassigning `position` (1-11 = starting XI in the given order, 12-15 = bench in the given
    order - caller decides bench order, e.g. bench GK last) and `multiplier` (0 = benched,
    otherwise 1, or 2 for whichever element is currently flagged is_captain). Does NOT touch
    is_captain/is_vice_captain - those are echoed back exactly as they currently are, so a
    lineup-only approval can't accidentally change the captain (that's approval_bot's separate
    'captain' kind). Raises ValueError if the current captain/vice-captain would end up benched -
    that's a real conflict between two independently-approved plans, not something to silently
    paper over with a guessed multiplier.

    starting_elements + bench_elements together must be exactly the 15 elements already in
    my_team["picks"] (no adds/removes - that's a transfer, submit_transfers' job).

    UNVERIFIED shape: only the narrower captain-only reassignment
    (element/position/multiplier/is_captain/is_vice_captain, position never changed) has been
    tested live and confirmed working (2026-08-26, scripts/test_captain_swap.py). Actually
    changing `position` is new. This endpoint has never shown the my-team-write danger the
    transfers endpoint has (no resource gets consumed by a POST here), so - unlike
    submit_transfers below - it's fine to verify with a real, small, reverted round-trip test
    (e.g. swap two bench outfield players' bench order, confirm via GET, swap back) before
    trusting it inside the unattended approval flow. Do that first if picking this up fresh.
    """
    current_captain = next(p["element"] for p in my_team["picks"] if p["is_captain"])
    current_vice = next(p["element"] for p in my_team["picks"] if p["is_vice_captain"])
    if current_captain not in starting_elements:
        raise ValueError(
            f"Current captain (element {current_captain}) is not in the proposed starting XI - "
            "resolve the captain approval before applying this lineup, don't guess."
        )
    if current_vice not in starting_elements:
        raise ValueError(
            f"Current vice-captain (element {current_vice}) is not in the proposed starting XI - "
            "resolve the captain approval before applying this lineup, don't guess."
        )

    position_by_element = {el: i + 1 for i, el in enumerate(starting_elements)}
    position_by_element.update({el: i + 12 for i, el in enumerate(bench_elements)})
    if set(position_by_element) != {p["element"] for p in my_team["picks"]}:
        raise ValueError(
            "starting_elements + bench_elements must be exactly the current 15 picks - "
            "this function only reorders/benches, it never adds or removes a player."
        )

    picks_payload = [
        {
            "element": p["element"],
            "position": position_by_element[p["element"]],
            "multiplier": 0 if p["element"] in bench_elements else (2 if p["element"] == current_captain else 1),
            "is_captain": p["is_captain"],
            "is_vice_captain": p["is_vice_captain"],
        }
        for p in my_team["picks"]
    ]
    url = MY_TEAM_URL.format(entry_id=entry_id)
    return session.post(url, json={"picks": picks_payload}, timeout=TIMEOUT_SECONDS)


def submit_transfers(
    session: requests.Session, entry_id: int, event_id: int, transfers: list[dict],
) -> requests.Response:
    """POSTs to /api/transfers/ with confirmed=true - a REAL, non-refundable submission.

    HARD RULE (project memory, from the 2026-08-26 incident where confirmed:false still
    executed a real transfer, and reverting it did NOT refund the free transfer it consumed):
    never call this function outside a real, deliberate, user-approved submission. There is no
    safe way to test it - `confirmed: false` is not a dry run in practice, and even a
    confirmed:true call that nets back to the original squad still permanently consumes a free
    transfer / can leave a real -4 hit. Never call this more than once per gameweek, and always
    pass the FULL set of transfers for that gameweek in one call - the real FPL web UI batches
    every edit in a session into one confirmed call representing the net change; calling this
    multiple times with partial transfer sets is exactly the mistake that caused the incident.

    transfers: list of {"element_in": int, "element_out": int, "purchase_price": int,
    "selling_price": int} - prices in FPL's *10 integer format (e.g. 45 = £4.5m), sourced from
    live now_cost values fetched as close to submission time as practical, not a stale plan
    snapshot (prices can move between when a plan was proposed and when it's approved).
    """
    payload = {
        "confirmed": True,
        "entry": entry_id,
        "event": event_id,
        "transfers": transfers,
        "wildcard": False,
        "freehit": False,
    }
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
