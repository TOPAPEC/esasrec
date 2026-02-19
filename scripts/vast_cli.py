from __future__ import annotations
import argparse
import os
import sys
import warnings


def main() -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--insecure", action="store_true")
    p.add_argument("--no-proxy", action="store_true")
    args, rest = p.parse_known_args()

    if args.no_proxy:
        for k in ["HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"]:
            os.environ.pop(k, None)

    if args.insecure:
        import requests
        import urllib3
        warnings.filterwarnings("ignore", category=urllib3.exceptions.InsecureRequestWarning)
        _send = requests.Session.send

        def _patched_send(self, request, **kwargs):
            kwargs["verify"] = False
            return _send(self, request, **kwargs)

        requests.Session.send = _patched_send

    import vast
    sys.argv = ["vastai", *rest]
    try:
        rc = vast.main()
        return int(rc or 0)
    except SystemExit as e:
        return int(e.code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
