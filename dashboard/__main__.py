import argparse
import os

import uvicorn

from .app import create_app


def main():
    parser = argparse.ArgumentParser(description="ArduBeeno event receiver and history dashboard")
    parser.add_argument("--host", default=os.environ.get("ARDUBEENO_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ARDUBEENO_PORT", "8000")))
    args = parser.parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
