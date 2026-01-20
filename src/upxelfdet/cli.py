"""Command-line interface for UPX ELF Detector.

This module provides a CLI for the UPX ELF Detector with enhanced logging support.

Usage:
    upx-elf-detector train --config config.json
    upx-elf-detector evaluate --config config.json
    upx-elf-detector predict --config config.json

Note:
    Logs are automatically saved to the directory specified in config.output.log
    (default: ./output/logs) with both human-readable and JSON formats.
"""

from maldet.cli import create_cli

from .detector import UpxElfDetector

# Create CLI application using the factory
app = create_cli(UpxElfDetector)

if __name__ == "__main__":
    app()
