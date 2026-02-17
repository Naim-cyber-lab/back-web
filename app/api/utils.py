"""
FastAPI endpoint: /api/v1/events/prefill_from_url_video

- Détecte la plateforme (YouTube / TikTok)
- Fait un "preview" best-effort
- NE DOIT PAS 502 pour des blocages YouTube/TikTok (anti-bot / challenge)
  => renvoie quand même un JSON avec des champs + meta_error si besoin

Prérequis:
- playwright installé + navigateurs: `playwright install`
- yt_dlp installé (optionnel: utilisé en best-effort)
- geopy installé
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

import yt_dlp
from playwright.async_api import async_playwright
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

router = APIRouter(prefix="/events", tags=["events"])

# =============================================================================
# PLATFORM DETECTION
# =============================================================================


def is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtu\.be|youtube\.com)", url or "", re.IGNORECASE))


def is_tiktok(url: str) -> bool:
    return bool(re.search(r"(tiktok\.com)", url or "", re.IGNORECASE))


# =============================================================================
# YOUTUBE HELPERS
# =============================================================================


def shorts_to_watch(url: str) -> str:
    """Convertit /shorts/<id> -> watch?v=<id> (plus stable pour comments + og tags)."""
    m = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", url or "")
    return f"https://www.youtube.com/watch?v={m.group(1)}" if m else url


def fetch_youtube_metadata_yt_dlp(url: str) -> Dict[str, Any]:
    """
    Best-effort via yt-dlp. Peut échouer à cause des challenges YouTube.
    On NE DOIT PAS laisser remonter l'exception au endpoint.
    """
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        # IMPORTANT: on ne veut pas télécharger/choisir des formats (souvent source d'erreurs)
        # extract_flat peut réduire les champs -> on garde "safe"
        "extract_flat": True,
        "nocheckcertificate": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    return {
        "title": info.get("title"),
        "description": info.get("description") or info.get("fulltitle") or "",
        "uploader": info.get("uploader") or info.get("channel"),
        "channel": info.get("channel"),
        "webpage_url": info.get("webpage_url") or url,
        "thumbnail": info.get("thumbnail"),
        "thumbnails": info.get("thumbnails") or [],
    }


async def fetch_youtube_og_tags(url: str, headless: bool = True) -> Dict[str, Any]:
    """
    Fallback robuste: og:title / og:description / og:image via Playwright.
    Ça marche souvent même quand yt-dlp se fait challenger.
    """
    watch = shorts_to_watch(url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(locale="fr-FR")
        page = await context.new_page()

        await page.goto(watch, wait_until="domcontentloaded")

        # consent best-effort
        for label in ["Tout accepter", "J'accepte", "Accept all", "I agree", "Agree"]:
            btn = page.get_by_role("button", name=re.compile(label, re.IGNORECASE))
            if await btn.count() > 0:
                try:
                    await btn.first.click(timeout=2000)
                    break
                except Exception:
                    pass

        # lire les OG tags
        og_title = await page.locator('meta[property="og:title"]').get_attribute("content")
        og_desc = await page.locator('meta[property="og:description"]').get_attribute("content")
        og_image = await page.locator('meta[property="og:image"]').get_attribute("content")

        await browser.close()

    return {
        "title": og_title,
        "description": og_desc or "",
        "thumbnail": og_image,
        "thumbnails": [{"url": og_image}] if og_image else [],
        "webpage_url": watch,
    }


async def fetch_first_comment_youtube(url: str, headless: bool = True) -> Optional[str]:
    """
    Best-effort: récupère le 1er commentaire visible.
    Peut retourner None (normal) si YouTube bloque / comments désactivés.
    """
    watch = shorts_to_watch(url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(locale="fr-FR")
        page = await context.new_page()

        await page.goto(watch, wait_until="domcontentloaded")

        # consent best-effort
        for label in ["Tout accepter", "J'accepte", "Accept all", "I agree", "Agree"]:
            btn = page.get_by_role("button", name=re.compile(label, re.IGNORECASE))
            if await btn.count() > 0:
                try:
                    await btn.first.click(timeout=2500)
                    break
                except Exception:
                    pass

        # scroller pour charger les commentaires
        for _ in range(12):
            await page.mouse.wheel(0, 1200)
            await page.wait_for_timeout(350)
            if await page.locator("ytd-comment-thread-renderer #content-text").count() > 0:
                break

        # parfois un popup gêne
        close_btn = page.locator('button[aria-label*="Fermer" i], button[aria-label*="Close" i]')
        if await close_btn.count() > 0:
            try:
                await close_btn.first.click(timeout=1200)
            except Exception:
                pass

        try:
            await page.wait_for_selector("ytd-comment-thread-renderer #content-text", timeout=6000)
        except Exception:
            await browser.close()
            return None

        first = page.locator("ytd-comment-thread-renderer #content-text").first
        txt = (await first.inner_text()).strip()

        await browser.close()
        return txt or None


# =============================================================================
# LOCATION / PRICE / RATING
# =============================================================================

def _infer_city_from_postal_code(cp: Optional[str]) -> Optional[str]:
    if not cp or not re.fullmatch(r"\d{5}", cp):
        return None
    if cp.startswith("75"):
        return "Paris"
    # (optionnel) autres villes
    if cp.startswith("69"):
        return "Lyon"
    if cp.startswith("13"):
        return "Marseille"
    return None


def _infer_city_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    t = text.lower()

    # hashtags
    for city in ["paris", "lyon", "marseille"]:
        if f"#{city}" in t:
            return city.capitalize()

    # "à Paris" / "a Paris"
    m = re.search(r"\b(?:à|a|sur|dans)\s+(paris|lyon|marseille)\b", t)
    if m:
        return m.group(1).capitalize()

    # simple présence du mot
    for city in ["paris", "lyon", "marseille"]:
        if re.search(rf"\b{city}\b", t):
            return city.capitalize()

    return None


def extract_fr_location(text: str) -> Optional[Dict[str, Optional[str]]]:
    if not text:
        return None

    t = text
    t = re.sub(r"\bAv\.\b", "avenue", t, flags=re.IGNORECASE)
    t = re.sub(r"\bBd\.\b", "boulevard", t, flags=re.IGNORECASE)
    t = re.sub(r"\bRte\.\b", "route", t, flags=re.IGNORECASE)
    t = re.sub(r"\bPl\.\b", "place", t, flags=re.IGNORECASE)
    t = re.sub(r"\bSt\.\b", "saint", t, flags=re.IGNORECASE)

    mcp = re.search(r"\b(?P<cp>\d{5})\b", t)
    cp = mcp.group("cp") if mcp else None

    city = None
    if mcp:
        after = t[mcp.end() :].strip()
        mcity = re.match(r"(?P<city>[A-Za-zÀ-ÿ'\- ]{2,60})", after)
        if mcity:
            city = re.split(r"[,\n|/]", mcity.group("city").strip())[0].strip()

    # ✅ fallback si city introuvable après CP (cas: "..., 75009")
    if not city:
        city = _infer_city_from_text(text) or _infer_city_from_postal_code(cp)

    addr_re = re.compile(
        r"""
        (?P<addr>
            \b\d{1,4}(?:\s*[-–]\s*\d{1,4})?
            \s*(?:bis|ter)?\s*
            (?:
                rue|avenue|boulevard|route|chemin|impasse|place|quai|cours|
                allée|allee|passage|sentier|voie|
                av\.?|bd\.?|rte\.?|pl\.?
            )
            \s+
            .*?
        )
        (?=
            \s+\d{5}\b
            |[,\n]
            |$
        )
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    maddr = addr_re.search(text) or addr_re.search(t)
    addr = maddr.group("addr").strip() if maddr else None
    if addr:
        addr = re.sub(r"\s+", " ", addr).strip(" -•|")

    if not (addr or cp or city):
        return None

    return {"address": addr, "postal_code": cp, "city": city}


def extract_price_info(text: str) -> Dict[str, Any]:
    if not text:
        return {"mentions": [], "best": None}

    t = text.replace("\u202f", " ").replace("\xa0", " ")
    for d in ["–", "—", "−", "‐", "‒"]:
        t = t.replace(d, "-")

    mentions: List[str] = []

    def to_float(s: str) -> float:
        return float(s.replace(",", "."))

    less_re = re.compile(
        r"""\b(?:moins\s+de|à\s+moins\s+de|<)\s*(?P<val>\d{1,3}(?:[.,]\d{1,2})?)\s*(?P<cur>€|eur|euros?)(?!\w)""",
        re.IGNORECASE | re.VERBOSE,
    )
    upto_re = re.compile(
        r"""\b(?:jusqu['’]?\s*à|jusqua|maximum|max|<=)\s*(?P<val>\d{1,3}(?:[.,]\d{1,2})?)\s*(?P<cur>€|eur|euros?)(?!\w)""",
        re.IGNORECASE | re.VERBOSE,
    )
    from_re = re.compile(
        r"""\b(?:à\s+partir\s+de|a\s+partir\s+de|dès|des|min(?:imum)?|>=)\s*(?P<val>\d{1,3}(?:[.,]\d{1,2})?)\s*(?P<cur>€|eur|euros?)(?!\w)""",
        re.IGNORECASE | re.VERBOSE,
    )
    between_re = re.compile(
        r"""\b(?:entre)\s*(?P<min>\d{1,3}(?:[.,]\d{1,2})?)\s*(?:et)\s*(?P<max>\d{1,3}(?:[.,]\d{1,2})?)\s*(?P<cur>€|eur|euros?)(?!\w)""",
        re.IGNORECASE | re.VERBOSE,
    )
    range_re = re.compile(
        r"""(?P<min>\d{1,3}(?:[.,]\d{1,2})?)\s*(?:-|à)\s*(?P<max>\d{1,3}(?:[.,]\d{1,2})?)\s*(?P<cur>€|eur|euros?)(?!\w)""",
        re.IGNORECASE | re.VERBOSE,
    )
    single_re = re.compile(
        r"""(?P<val>\d{1,3}(?:[.,]\d{1,2})?)\s*(?P<cur>€|eur|euros?)(?!\w)""",
        re.IGNORECASE | re.VERBOSE,
    )
    euro_scale_re = re.compile(r"(?<!\w)(€{1,4})(?!\w)")

    for rx in [less_re, upto_re, from_re, between_re, range_re, single_re, euro_scale_re]:
        for m in rx.finditer(t):
            mentions.append(m.group(0).strip())

    best = None
    m = between_re.search(t)
    if m:
        best = {
            "type": "range",
            "min": to_float(m.group("min")),
            "max": to_float(m.group("max")),
            "currency": "EUR",
            "raw": m.group(0).strip(),
            "operator": "between",
        }
    else:
        m = range_re.search(t)
        if m:
            best = {
                "type": "range",
                "min": to_float(m.group("min")),
                "max": to_float(m.group("max")),
                "currency": "EUR",
                "raw": m.group(0).strip(),
                "operator": "range",
            }
        else:
            m = less_re.search(t) or upto_re.search(t)
            if m:
                best = {
                    "type": "range",
                    "min": 0.0,
                    "max": to_float(m.group("val")),
                    "currency": "EUR",
                    "raw": m.group(0).strip(),
                    "operator": "max",
                }
            else:
                m = from_re.search(t)
                if m:
                    best = {
                        "type": "range",
                        "min": to_float(m.group("val")),
                        "max": None,
                        "currency": "EUR",
                        "raw": m.group(0).strip(),
                        "operator": "min",
                    }
                else:
                    ms = euro_scale_re.search(t)
                    if ms:
                        best = {"type": "scale", "scale": ms.group(1), "raw": ms.group(1)}
                    else:
                        m2 = single_re.search(t)
                        if m2:
                            best = {
                                "type": "single",
                                "value": to_float(m2.group("val")),
                                "currency": "EUR",
                                "raw": m2.group(0).strip(),
                            }

    uniq, seen = [], set()
    for x in mentions:
        if x and x not in seen:
            uniq.append(x)
            seen.add(x)

    return {"mentions": uniq, "best": best}


def extract_rating_info(text: str) -> Dict[str, Any]:
    if not text:
        return {"mentions": [], "best": None}

    t = text.replace("\u202f", " ").replace("\xa0", " ")
    mentions: List[str] = []

    frac_re = re.compile(r"\b(?P<score>[0-5](?:[.,]\d)?)\s*/\s*(?P<outof>5)\b")
    for m in frac_re.finditer(t):
        mentions.append(m.group(0))

    star_re = re.compile(r"\b(?P<score>[0-5](?:[.,]\d)?)\s*(?:étoiles?|etoiles?|stars?|⭐)\b", re.IGNORECASE)
    for m in star_re.finditer(t):
        mentions.append(m.group(0))

    best = None
    m = frac_re.search(t)
    if m:
        best = {"score": float(m.group("score").replace(",", ".")), "out_of": 5, "raw": m.group(0)}
    else:
        m2 = star_re.search(t)
        if m2:
            best = {"score": float(m2.group("score").replace(",", ".")), "out_of": 5, "raw": m2.group(0)}

    uniq, seen = [], set()
    for x in mentions:
        if x not in seen:
            uniq.append(x)
            seen.add(x)

    return {"mentions": uniq, "best": best}


# =============================================================================
# GEOCODING (cached)
# =============================================================================

_geolocator = Nominatim(user_agent="social-location-extractor")
_geocode = RateLimiter(_geolocator.geocode, min_delay_seconds=1.0, swallow_exceptions=True)
_geocode_cache: Dict[str, Dict[str, Any]] = {}


def geocode_location(loc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not loc:
        return None

    addr = loc.get("address")
    cp = loc.get("postal_code")
    city = loc.get("city")

    parts = [p for p in [addr, cp, city, "France"] if p]
    query = ", ".join(parts).strip()
    if not query:
        return loc

    if query in _geocode_cache:
        return {**loc, **_geocode_cache[query]}

    result = _geocode(query, addressdetails=True, language="fr")
    if not result:
        return loc

    raw = result.raw or {}
    address = raw.get("address", {}) if isinstance(raw, dict) else {}
    region = address.get("state") or address.get("region") or address.get("county")
    country = address.get("country")

    # ✅ city depuis Nominatim si manquante
    city_geo = None
    for k in ["city", "town", "village", "municipality", "hamlet"]:
        if address.get(k):
            city_geo = address[k]
            break

    # fallback via display_name (utile pour "Paris 9e Arrondissement")
    if not city_geo and isinstance(raw.get("display_name"), str):
        m = re.search(r"\b(Paris|Lyon|Marseille)\b", raw["display_name"])
        if m:
            city_geo = m.group(1)

    enriched = {
        "latitude": float(result.latitude),
        "longitude": float(result.longitude),
        "region": region,
        "country": country,
        "display_name": raw.get("display_name"),

        # ✅ on force city si loc.city est vide
        "city": city or city_geo,
        # ✅ on complète CP si manquant
        "postal_code": cp or address.get("postcode"),
    }


    _geocode_cache[query] = enriched
    return {**loc, **enriched}


# =============================================================================
# TIKTOK HELPERS
# =============================================================================


def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _find_in_obj(obj: Any, key: str) -> List[Any]:
    out: List[Any] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out.extend(_find_in_obj(v, key))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_find_in_obj(it, key))
    return out


async def fetch_tiktok_metadata(url: str, headless: bool = True) -> Dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(locale="fr-FR")
        page = await context.new_page()

        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)

        # consent best-effort
        for label in ["Tout accepter", "Accepter", "Accept all", "I agree", "Agree"]:
            btn = page.get_by_role("button", name=re.compile(label, re.IGNORECASE))
            if await btn.count() > 0:
                try:
                    await btn.first.click(timeout=2000)
                    break
                except Exception:
                    pass

        await page.wait_for_timeout(1200)

        og_title = await page.locator('meta[property="og:title"]').get_attribute("content")
        og_desc = await page.locator('meta[property="og:description"]').get_attribute("content")
        og_image = await page.locator('meta[property="og:image"]').get_attribute("content")

        sigi = None
        uni = None
        sigi_el = page.locator("#SIGI_STATE")
        if await sigi_el.count() > 0:
            sigi = _safe_json_loads((await sigi_el.first.inner_text()) or "")

        uni_el = page.locator("#__UNIVERSAL_DATA_FOR_REHYDRATION__")
        if await uni_el.count() > 0:
            uni = _safe_json_loads((await uni_el.first.inner_text()) or "")

        caption = None
        author = None
        thumbnails: List[Dict[str, Any]] = []

        for blob in [sigi, uni]:
            if not blob:
                continue

            cands = _find_in_obj(blob, "desc") + _find_in_obj(blob, "description") + _find_in_obj(blob, "caption")
            for c in cands:
                if isinstance(c, str) and len(c.strip()) >= 3:
                    if caption is None or len(c) > len(caption):
                        caption = c.strip()

            acands = _find_in_obj(blob, "uniqueId") + _find_in_obj(blob, "nickname")
            for a in acands:
                if isinstance(a, str) and 2 <= len(a.strip()) <= 40 and author is None:
                    author = a.strip()

            url_fields: List[Any] = []
            for key in ["cover", "dynamicCover", "originCover", "poster", "thumbnail", "avatarThumb"]:
                url_fields.extend(_find_in_obj(blob, key))

            def add_thumb(u: Any):
                if isinstance(u, str) and u.startswith("http"):
                    thumbnails.append({"url": u})
                elif isinstance(u, list):
                    for it in u:
                        if isinstance(it, str) and it.startswith("http"):
                            thumbnails.append({"url": it})
                elif isinstance(u, dict):
                    for it in u.values():
                        if isinstance(it, str) and it.startswith("http"):
                            thumbnails.append({"url": it})

            for u in url_fields:
                add_thumb(u)

        # dedupe
        norm_thumbs: List[Dict[str, Any]] = []
        seen = set()
        for t in thumbnails:
            u = t.get("url")
            if not u or u in seen:
                continue
            seen.add(u)
            norm_thumbs.append({"url": u})

        thumbnail = og_image or (norm_thumbs[0]["url"] if norm_thumbs else None)
        title = caption or og_title or None
        description = caption or og_desc or ""

        await browser.close()

        return {
            "title": title,
            "description": description,
            "uploader": author,
            "webpage_url": url,
            "thumbnail": thumbnail,
            "thumbnails": norm_thumbs,
        }


async def fetch_first_comment_tiktok(url: str, headless: bool = True) -> Optional[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(locale="fr-FR")
        page = await context.new_page()

        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(2000)

        # consent best-effort
        for label in ["Tout accepter", "Accepter", "Accept all", "I agree", "Agree"]:
            btn = page.get_by_role("button", name=re.compile(label, re.IGNORECASE))
            if await btn.count() > 0:
                try:
                    await btn.first.click(timeout=2000)
                    break
                except Exception:
                    pass

        await page.wait_for_timeout(1500)

        for _ in range(6):
            await page.mouse.wheel(0, 900)
            await page.wait_for_timeout(450)

        candidates = [
            page.locator('[data-e2e*="comment"] p').first,
            page.locator('div[class*="DivCommentItemContainer"] p').first,
            page.locator('div[class*="Comment"] p').first,
        ]

        txt = None
        for loc in candidates:
            try:
                if await loc.count() > 0:
                    v = (await loc.inner_text()).strip()
                    if v:
                        txt = v
                        break
            except Exception:
                pass

        await browser.close()
        return txt or None


# =============================================================================
# ORCHESTRATION (best-effort, never raises for external blockers)
# =============================================================================


async def preview_youtube(url: str, headless: bool) -> Dict[str, Any]:
    started = time.time()
    meta_error: Optional[str] = None

    # 1) try yt-dlp (best effort)
    meta: Dict[str, Any] = {}
    try:
        meta = fetch_youtube_metadata_yt_dlp(url)
    except Exception as e:
        meta_error = f"yt-dlp failed: {type(e).__name__}: {e}"
        meta = {}

    # 2) fallback og tags if needed
    if not meta.get("title") and not meta.get("description") and not meta.get("thumbnail"):
        try:
            og = await fetch_youtube_og_tags(url, headless=headless)
            # merge og into meta (without destroying existing)
            meta = {**og, **meta}
        except Exception as e:
            # keep going anyway
            meta_error = (meta_error + " | " if meta_error else "") + f"og-tags failed: {type(e).__name__}: {e}"

    # 3) comments (best effort)
    first_comment = None
    try:
        first_comment = await fetch_first_comment_youtube(url, headless=headless)
    except Exception as e:
        meta_error = (meta_error + " | " if meta_error else "") + f"comments failed: {type(e).__name__}: {e}"

    desc_text = (meta.get("description") or "") if isinstance(meta, dict) else ""
    comment_text = first_comment or ""

    loc_from_desc = extract_fr_location(desc_text)
    loc_from_comment = extract_fr_location(comment_text)
    best_loc = loc_from_desc or loc_from_comment
    best_loc_geocoded = geocode_location(best_loc) if best_loc else None

    return {
        "platform": "YouTube",
        "url": url,
        "watch_url_used_for_comments": shorts_to_watch(url),

        "title": meta.get("title"),
        "description": desc_text,

        "thumbnail": meta.get("thumbnail"),
        "thumbnails": meta.get("thumbnails") or [],

        "uploader": meta.get("uploader"),
        "channel": meta.get("channel"),
        "webpage_url": meta.get("webpage_url") or url,

        "first_comment": first_comment,

        "location_from_description": loc_from_desc,
        "location_from_first_comment": loc_from_comment,
        "best_location": best_loc,
        "best_location_geocoded": best_loc_geocoded,

        "price_from_description": extract_price_info(desc_text),
        "price_from_first_comment": extract_price_info(comment_text),

        "rating_from_description": extract_rating_info(desc_text),
        "rating_from_first_comment": extract_rating_info(comment_text),

        # diagnostics non bloquants
        "meta_error": meta_error,
        "elapsed_ms": int((time.time() - started) * 1000),
    }


async def preview_tiktok(url: str, headless: bool) -> Dict[str, Any]:
    started = time.time()
    meta_error: Optional[str] = None

    meta: Dict[str, Any] = {}
    try:
        meta = await fetch_tiktok_metadata(url, headless=headless)
    except Exception as e:
        meta_error = f"tiktok meta failed: {type(e).__name__}: {e}"
        meta = {}

    first_comment = None
    try:
        first_comment = await fetch_first_comment_tiktok(url, headless=headless)
    except Exception as e:
        meta_error = (meta_error + " | " if meta_error else "") + f"tiktok comments failed: {type(e).__name__}: {e}"

    desc_text = (meta.get("description") or "") if isinstance(meta, dict) else ""
    comment_text = first_comment or ""

    loc_from_desc = extract_fr_location(desc_text)
    loc_from_comment = extract_fr_location(comment_text)
    best_loc = loc_from_desc or loc_from_comment
    best_loc_geocoded = geocode_location(best_loc) if best_loc else None

    return {
        "platform": "TikTok",
        "url": url,

        "title": meta.get("title"),
        "description": desc_text,

        "thumbnail": meta.get("thumbnail"),
        "thumbnails": meta.get("thumbnails") or [],

        "uploader": meta.get("uploader"),
        "webpage_url": meta.get("webpage_url") or url,

        "first_comment": first_comment,

        "location_from_description": loc_from_desc,
        "location_from_first_comment": loc_from_comment,
        "best_location": best_loc,
        "best_location_geocoded": best_loc_geocoded,

        "price_from_description": extract_price_info(desc_text),
        "price_from_first_comment": extract_price_info(comment_text),

        "rating_from_description": extract_rating_info(desc_text),
        "rating_from_first_comment": extract_rating_info(comment_text),

        # diagnostics non bloquants
        "meta_error": meta_error,
        "elapsed_ms": int((time.time() - started) * 1000),
    }



