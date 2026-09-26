#!/usr/bin/env python3
"""Everything `boltz` does at the command line, done from Python.

    export BOLTZLABS_API_KEY=ak_...      # or put it in a .env next to this file
    python examples/sandbox_tour.py

It creates a real sandbox, which costs real money for as long as it runs — so it
is created inside a `with`, and destroyed even if this script raises.
"""

import argparse

import boltzlabs
from boltzlabs import Sandbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="override the origin (default: production)")
    ap.add_argument("--api-key", help="override the key (default: env or .env)")
    ap.add_argument("--environment", default="python", help="runtime or coding-agent image")
    ap.add_argument("--machine", default="small", help="small, medium, large")
    args = ap.parse_args()

    if args.url or args.api_key:
        boltzlabs.use(api_key=args.api_key, url=args.url)

    print(f"logged in as {boltzlabs.me().get('email')}")

    print("\nenvironments ", ", ".join(str(e) for e in boltzlabs.environments()))
    print("machines     ", ", ".join(f"{m.name} (${m.rate_usd_per_hour:.2f}/hr)" for m in boltzlabs.machines()))

    print("\nyour sandboxes")
    for sb in boltzlabs.sandboxes():
        print(f"  {sb.id}  {sb.status:<9} {sb.environment:<8} {sb.runtime_label:>6}  ${sb.cost_usd:.2f}")

    print(f"\ncreating a {args.environment} sandbox…")
    with Sandbox(environment=args.environment, machine=args.machine, name="sdk-tour") as sb:
        print(f"  {sb.id} — ${sb.rate_usd_per_hour:.2f}/hr")

        print("\nexec")
        print("  " + str(sb.exec("uname -a && python3 -V")).strip().replace("\n", "\n  "))

        print("\nrun")
        print("  " + str(sb.run("print(sum(range(101)))")).strip())

        print("\nterminal (a real PTY, like `boltz connect`)")
        print("  " + sb.terminal("tty; echo from-a-real-tty").strip().replace("\n", "\n  ")[:400])

        print(f"\nport 8080 is at {sb.url(8080)}")

    print("\nsandbox destroyed")


if __name__ == "__main__":
    main()
