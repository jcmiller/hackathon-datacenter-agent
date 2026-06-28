"""
Computer use session: Gemini 3.5 Flash with ComputerUse tool + Playwright.

Flow per turn:
  1. Take a Playwright screenshot of the dashboard
  2. Send screenshot + task to gemini-3.5-flash with ComputerUse(ENVIRONMENT_BROWSER)
  3. Parse returned function_call parts as UI actions
  4. Execute each action on the live browser page
  5. Yield SSE-compatible event dicts; repeat for up to MAX_TURNS
"""

import asyncio
import base64
import os
from collections.abc import AsyncGenerator

from google.genai import Client, types

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8000")
MODEL = "gemini-3.5-flash"
MAX_TURNS = 5
VIEWPORT = {"width": 1280, "height": 800}

DEFAULT_TASK = (
    "You are an expert GPU fleet reliability engineer. "
    "This is the GPUSitter monitoring dashboard. "
    "1. Identify the most critical GPU incident shown in the incident feed on the left. "
    "2. Click on it to select it. "
    "3. Wait for the AI triage panel on the right to show the agent reasoning. "
    "4. Report what disposition the agent reached and why. "
    "Be precise — click exactly on the incident row."
)

_COMPUTER_USE_TOOL = types.Tool(
    computer_use=types.ComputerUse(environment=types.Environment.ENVIRONMENT_BROWSER)
)

_CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "You are controlling a real browser showing a GPU fleet monitoring dashboard. "
        "Take actions to investigate and triage the most critical GPU incident. "
        "After each action, a new screenshot will be provided. "
        "Describe what you see and what you are doing step by step."
    ),
    tools=[_COMPUTER_USE_TOOL],
    temperature=0,
)


async def _screenshot(page) -> tuple[bytes, str]:
    """Take a Playwright screenshot; return raw bytes and base64 string."""
    raw = await page.screenshot(type="png")
    return raw, base64.b64encode(raw).decode()


def _image_part(raw: bytes) -> types.Part:
    return types.Part(inline_data=types.Blob(mime_type="image/png", data=raw))


async def _execute_action(page, name: str, args: dict) -> str:
    """Execute one computer-use action on the playwright page."""
    try:
        if name in ("click", "left_click", "single_click"):
            coord = args.get("coordinate") or [args.get("x", 0), args.get("y", 0)]
            await page.mouse.click(float(coord[0]), float(coord[1]))
            return f"clicked ({coord[0]}, {coord[1]})"

        if name in ("right_click",):
            coord = args.get("coordinate") or [args.get("x", 0), args.get("y", 0)]
            await page.mouse.click(float(coord[0]), float(coord[1]), button="right")
            return f"right-clicked ({coord[0]}, {coord[1]})"

        if name in ("double_click",):
            coord = args.get("coordinate") or [args.get("x", 0), args.get("y", 0)]
            await page.mouse.dblclick(float(coord[0]), float(coord[1]))
            return f"double-clicked ({coord[0]}, {coord[1]})"

        if name in ("type", "type_text", "input_text"):
            text = args.get("text", "")
            await page.keyboard.type(text)
            return f"typed: {text!r}"

        if name in ("key", "key_stroke", "press"):
            key = args.get("key", "") or args.get("text", "")
            await page.keyboard.press(key)
            return f"pressed: {key}"

        if name in ("scroll",):
            coord = args.get("coordinate") or [args.get("x", 640), args.get("y", 400)]
            direction = args.get("direction", "down")
            amount = int(args.get("amount", 3))
            delta = 120 * amount * (1 if direction == "down" else -1)
            await page.mouse.wheel(float(coord[0]), float(coord[1]), 0, delta)
            return f"scrolled {direction} at ({coord[0]}, {coord[1]})"

        if name in ("drag", "drag_and_drop"):
            start = args.get("start_coordinate") or [args.get("start_x", 0), args.get("start_y", 0)]
            end = args.get("end_coordinate") or [args.get("end_x", 0), args.get("end_y", 0)]
            await page.mouse.move(float(start[0]), float(start[1]))
            await page.mouse.down()
            await page.mouse.move(float(end[0]), float(end[1]))
            await page.mouse.up()
            return f"dragged ({start[0]},{start[1]}) → ({end[0]},{end[1]})"

        if name in ("screenshot", "take_screenshot", "capture_screenshot"):
            return "screenshot taken"

        if name in ("wait", "pause"):
            ms = int(args.get("ms", args.get("duration", 1000)))
            await asyncio.sleep(ms / 1000)
            return f"waited {ms}ms"

        # Unknown action — acknowledge but don't fail
        return f"unhandled action: {name}"
    except Exception as exc:
        return f"action error: {exc}"


async def run_session(
    task: str | None = None,
) -> AsyncGenerator[dict]:
    """
    Yield SSE-compatible event dicts for one computer-use session.

    Event types:
      screenshot  — {"type":"screenshot","data":<b64>,"turn":<n>}
      reasoning   — {"type":"reasoning","text":<str>,"turn":<n>}
      action      — {"type":"action","name":<str>,"args":<dict>,"result":<str>,"turn":<n>}
      done        — {"type":"done","turns":<n>}
      error       — {"type":"error","message":<str>}
    """
    from playwright.async_api import async_playwright

    if task is None:
        task = DEFAULT_TASK

    client = Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page(viewport=VIEWPORT)

            # Load the dashboard — use domcontentloaded; SSE keeps connection open so
            # networkidle never fires.
            await page.goto(DASHBOARD_URL, wait_until="domcontentloaded", timeout=20_000)
            # Wait for incident rows to stream in
            try:
                await page.wait_for_selector(".inc", timeout=20_000)
                await asyncio.sleep(2)  # let a few more incidents arrive
            except Exception:
                await asyncio.sleep(5)

            raw, b64 = await _screenshot(page)
            yield {"type": "screenshot", "data": b64, "turn": 0}

            # Build initial message
            contents: list[types.Content] = [
                types.Content(
                    role="user",
                    parts=[_image_part(raw), types.Part(text=task)],
                )
            ]

            for turn in range(1, MAX_TURNS + 1):
                response = client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=_CONFIG,
                )

                candidate = response.candidates[0]
                tool_result_parts: list[types.Part] = []
                acted = False

                for part in candidate.content.parts:
                    if part.text:
                        yield {"type": "reasoning", "text": part.text, "turn": turn}

                    if part.function_call:
                        name = part.function_call.name
                        args = dict(part.function_call.args or {})

                        result = await _execute_action(page, name, args)
                        acted = True
                        yield {
                            "type": "action",
                            "name": name,
                            "args": args,
                            "result": result,
                            "turn": turn,
                        }

                        # Take fresh screenshot for screenshot actions or after any click
                        if name in (
                            "screenshot",
                            "take_screenshot",
                            "capture_screenshot",
                            "click",
                            "left_click",
                            "single_click",
                        ):
                            await asyncio.sleep(1.5)  # let UI settle
                            raw, b64 = await _screenshot(page)
                            yield {"type": "screenshot", "data": b64, "turn": turn}

                        tool_result_parts.append(
                            types.Part(
                                function_response=types.FunctionResponse(
                                    name=name,
                                    response={"result": result},
                                )
                            )
                        )

                # Append assistant turn
                contents.append(candidate.content)

                finish = getattr(candidate.finish_reason, "name", str(candidate.finish_reason))
                if finish == "STOP" and not acted:
                    break

                if tool_result_parts:
                    # Take a fresh screenshot to give the model an updated view
                    await asyncio.sleep(2)
                    raw, b64 = await _screenshot(page)
                    yield {"type": "screenshot", "data": b64, "turn": turn}
                    tool_result_parts.append(_image_part(raw))
                    contents.append(types.Content(role="user", parts=tool_result_parts))
                else:
                    break

            await browser.close()

    except Exception as exc:
        yield {"type": "error", "message": str(exc)}
        return

    yield {"type": "done", "turns": turn}
