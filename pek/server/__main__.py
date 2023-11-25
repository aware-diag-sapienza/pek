import argparse

from .server import PEKServer


def main(args):
    server = PEKServer(args.port)
    server.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="pek.server")
    parser.add_argument("-p", "--port", help="port to listen for connections", default=9786)
    parser.add_argument("-verbose", "--verbose", help="print debug information", action="store_true")
    main(parser.parse_args())
