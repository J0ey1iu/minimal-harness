import sys

from .tui import ChatTUI


def main():
    args = sys.argv[1:]
    if not args or args[0] == "chat":
        app = ChatTUI()
        app.run()
    else:
        print(f"Unknown command: {args[0]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
