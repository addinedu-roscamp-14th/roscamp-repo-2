"""Backward-compatible executable alias for pinky_tcp_bridge."""

from central_control.pinky_tcp_bridge import PinkyTcpBridge, main

PinkyTcpAdapter = PinkyTcpBridge

__all__ = ["PinkyTcpAdapter", "PinkyTcpBridge", "main"]

if __name__ == "__main__":
    main()
