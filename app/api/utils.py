"""
FastAPI endpoint: /api/v1/events/from-social/preview

- Detecte la plateforme (YouTube / TikTok)
- Appelle les fonctions adaptées
- Retourne un JSON "preview" standardisé utilisable côté front

⚠️ Prérequis:
- playwright installé + navigateurs: `playwright install`
- yt_dlp installé
- geopy installé
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse


# =============================================================================
# PLATFORM DETECTION
# =============================================================================

def is_youtube(url: str) -> bool:
    return bool(re.search(r"(youtu\.be|youtube\.com)", url or "", re.IGNORECASE))


def is_tiktok(url: str) -> bool:
    return bool(re.search(r"(tiktok\.com)", url or "", re.IGNORECASE))


# =============================================================================
# IMPORT YOUR EXISTING FUNCTIONS
# (I paste them as-is / minimal edits for type hints compatibility)
# =============================================================================

import json
import yt_dlp
from playwright.async_api import async_playwright

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


# -------------------- yt-dlp: title + description + thumbnails --------------------
def fetch_youtube_metadata(url: str):
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,

        # ✅ clé : ne récupère pas les formats -> évite pas mal d'erreurs
        "extract_flat": True,

        # optionnel mais utile
        "nocheckcertificate": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    # extract_flat renvoie parfois moins de champs => fallback safe
    return {
        "title": info.get("title"),
        "description": info.get("description") or info.get("fulltitle") or "",
        "uploader": info.get("uploader") or info.get("channel"),
        "channel": info.get("channel"),
        "webpage_url": info.get("webpage_url") or url,
        "thumbnail": info.get("thumbnail"),
        "thumbnails": info.get("thumbnails") or [],
    }

# -------------------- location extraction (FR heuristic) --------------------
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


# -------------------- price extraction (mentions + best estimate) --------------------
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


# -------------------- rating extraction (mentions + best) --------------------
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


# -------------------- helper: shorts -> watch URL --------------------
def shorts_to_watch(url: str) -> str:
    m = re.search(r"/shorts/([A-Za-z0-9_-]{6,})", url)
    return f"https://www.youtube.com/watch?v={m.group(1)}" if m else url


# -------------------- Playwright: first visible comment (YouTube) --------------------
async def fetch_first_comment_youtube(url: str, headless: bool = True) -> Optional[str]:
    url = shorts_to_watch(url)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(locale="fr-FR")
        page = await context.new_page()

        await page.goto(url, wait_until="domcontentloaded")

        for label in ["Tout accepter", "J'accepte", "Accept all", "I agree", "Agree"]:
            btn = page.get_by_role("button", name=re.compile(label, re.IGNORECASE))
            if await btn.count() > 0:
                try:
                    await btn.first.click(timeout=2500)
                    break
                except Exception:
                    pass

        for _ in range(12):
            await page.mouse.wheel(0, 1200)
            await page.wait_for_timeout(400)
            if await page.locator("ytd-comment-thread-renderer #content-text").count() > 0:
                break

        close_btn = page.locator('button[aria-label*="Fermer" i], button[aria-label*="Close" i]')
        if await close_btn.count() > 0:
            try:
                await close_btn.first.click(timeout=1200)
            except Exception:
                pass

        try:
            await page.wait_for_selector("ytd-comment-thread-renderer #content-text", timeout=8000)
        except Exception:
            await browser.close()
            return None

        first = page.locator("ytd-comment-thread-renderer #content-text").first
        txt = (await first.inner_text()).strip()

        await browser.close()
        return txt or None


# -------------------- Geocoding: lat/lon + région + pays --------------------
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

    enriched = {
        "latitude": float(result.latitude),
        "longitude": float(result.longitude),
        "region": region,
        "country": country,
        "display_name": raw.get("display_name"),
    }

    _geocode_cache[query] = enriched
    return {**loc, **enriched}


# -------------------- TikTok helpers --------------------
def _safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None


def _find_in_obj(obj: Any, key: str) -> List[Any]:
    out = []
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
                if isinstance(a, str) and 2 <= len(a.strip()) <= 40:
                    if author is None:
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

        norm_thumbs = []
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
# ORCHESTRATION
# =============================================================================

async def preview_youtube(url: str, headless: bool) -> Dict[str, Any]:
    meta = fetch_youtube_metadata(url)
    first_comment = await fetch_first_comment_youtube(url, headless=headless)

    desc_text = meta.get("description") or ""
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
        "webpage_url": meta.get("webpage_url"),

        "first_comment": first_comment,

        "location_from_description": loc_from_desc,
        "location_from_first_comment": loc_from_comment,
        "best_location": best_loc,
        "best_location_geocoded": best_loc_geocoded,

        "price_from_description": extract_price_info(desc_text),
        "price_from_first_comment": extract_price_info(comment_text),

        "rating_from_description": extract_rating_info(desc_text),
        "rating_from_first_comment": extract_rating_info(comment_text),
    }


async def preview_tiktok(url: str, headless: bool) -> Dict[str, Any]:
    meta = await fetch_tiktok_metadata(url, headless=headless)
    first_comment = await fetch_first_comment_tiktok(url, headless=headless)

    desc_text = meta.get("description") or ""
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
        "webpage_url": meta.get("webpage_url"),

        "first_comment": first_comment,

        "location_from_description": loc_from_desc,
        "location_from_first_comment": loc_from_comment,
        "best_location": best_loc,
        "best_location_geocoded": best_loc_geocoded,

        "price_from_description": extract_price_info(desc_text),
        "price_from_first_comment": extract_price_info(comment_text),

        "rating_from_description": extract_rating_info(desc_text),
        "rating_from_first_comment": extract_rating_info(comment_text),
    }



