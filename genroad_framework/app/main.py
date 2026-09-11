"""Command-line entry point for the GenRoad Framework web application."""

from __future__ import annotations

import argparse

from app.ui import create_interface, initialize_app


def build_parser() -> argparse.ArgumentParser:
    """Build the web application argument parser."""
    parser = argparse.ArgumentParser(
        description="Launch the GenRoad Framework web application."
    )
    parser.add_argument("--port", type=int, default=7860, help="Local HTTP port.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. Use 0.0.0.0 only behind appropriate access controls.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a temporary Gradio share link. Do not use for sensitive images.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a user-created YAML configuration file.",
    )
    return parser


def main() -> None:
    """Initialize and launch the Gradio application."""
    args = build_parser().parse_args()
    initialize_app(args.config)
    demo = create_interface()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
