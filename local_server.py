"""Simple local HTTP server for Rosetta Bear artifacts."""

from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import os


def run_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Serve the repository root so docs and specs can be browsed locally."""
    os.chdir(os.path.dirname(__file__))
    handler = SimpleHTTPRequestHandler
    with TCPServer((host, port), handler) as httpd:
        print(f"Serving Rosetta Bear repo at http://{host}:{port}")
        httpd.serve_forever()


if __name__ == "__main__":
    run_server(
        host=os.environ.get("ROSETTA_SERVER_HOST", "127.0.0.1"),
        port=int(os.environ.get("ROSETTA_SERVER_PORT", "8080")),
    )
