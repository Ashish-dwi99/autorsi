"""Manual two-case DEMONSTRATION entrypoint; never consumes verification cases.

From the repo with the installed Playwright interpreter:
  python proposals/adam_screen_v1/screen_cli.py run --run-dir /absolute/new-dir \
    --session-id unique-session --actor-label 'Actual assistant identity'
  python proposals/adam_screen_v1/screen_cli.py next --run-dir /absolute/new-dir
Inspect the returned screenshot with an image-view tool, then submit coordinates:
  python proposals/adam_screen_v1/screen_cli.py submit --run-dir /absolute/new-dir \
    --response '{"actions":[{"click":{"x":100,"y":200}}]}'
No DOM/source inspection is part of the actor interface. Root must retain actual
actor model/tool provenance. Interactive elapsed time includes tool/coordination
delay and is not comparable with an API-only inference latency measurement.
"""
import argparse
import asyncio
import json
from pathlib import Path

from screen_cases import demo_cases
from screen_harness import ScreenSession, write_json
from screen_manual import ManualPolicy, next_observation, submit
from screen_protocol import Limits


async def run(args):
    from playwright.async_api import async_playwright
    folder = Path(args.run_dir).resolve()
    folder.mkdir(parents=False, exist_ok=False)
    policy = ManualPolicy(folder / "bridge", args.session_id)
    session = ScreenSession(policy, actor_label=args.actor_label,
                            actor_identity={"declared_by_caller": args.actor_label,
                                            "verification": "External runtime provenance required; CLI does not verify model identity",
                                            "interface": "Screenshot plus typed action file; tool/coordination delay included"},
                            limits=Limits(call_timeout_s=60))
    report = {"schema": 1, "kind": "two-case-development-demonstration", "actor_label": args.actor_label,
              "session_id": args.session_id, "planned_cases": [c.case_id for c in demo_cases()],
              "verification_cases_used": 0, "rows": [], "browser_close_returned": False}
    write_json(folder / "demo.json", report)
    async with async_playwright() as pw:
        launch = {"headless": True}
        if args.browser_executable:
            binary = Path(args.browser_executable).resolve()
            if not binary.is_file():
                raise ValueError("explicit browser executable does not exist")
            launch["executable_path"] = str(binary)
            report["browser_executable"] = str(binary)
        browser = await pw.chromium.launch(**launch)
        try:
            for case in demo_cases():
                policy.case_id = case.case_id
                row = await session.run_case(browser, case, folder / case.case_id, demonstration=True)
                report["rows"].append(row)
                write_json(folder / "demo.json", report)
        finally:
            await asyncio.wait_for(browser.close(), 10)
            report["browser_close_returned"] = True
            write_json(folder / "demo.json", report)
    print(json.dumps({"report": str(folder / "demo.json"), "demonstration_only": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    start = sub.add_parser("run")
    start.add_argument("--run-dir", required=True)
    start.add_argument("--session-id", required=True)
    start.add_argument("--actor-label", required=True)
    start.add_argument("--browser-executable", help="Optional explicitly chosen installed browser; fresh Playwright profile only")
    for name in ("next", "submit"):
        command = sub.add_parser(name)
        command.add_argument("--run-dir", required=True)
        if name == "submit":
            command.add_argument("--response", required=True)
    args = parser.parse_args()
    if args.command == "run":
        asyncio.run(run(args))
    elif args.command == "next":
        print(json.dumps(next_observation(Path(args.run_dir) / "bridge"), indent=2))
    else:
        print(json.dumps(submit(Path(args.run_dir) / "bridge", args.response)))


if __name__ == "__main__":
    main()
