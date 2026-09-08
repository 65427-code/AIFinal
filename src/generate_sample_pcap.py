"""
Synthetic PCAP Generator for AI-Based Threat Detection
Generates realistic multi-flow network traffic containing both benign and attack flows:
1. Benign HTTPS (Full TLS 1.3 encrypted web session: handshake, requests, multi-packet responses, teardown)
2. Benign DNS (Standard UDP domain lookup query and response)
3. PortScan Reconnaissance (Probing ports with characteristic PSH scanner packets and target responses)
4. DoS Hulk / Slowloris Attack (High-speed TCP bursts to web port 80 with DoS signature)
5. DDoS Application Flood (Multi-packet distributed flood pattern directed at port 80)
"""

import os
from scapy.all import wrpcap, IP, TCP, UDP, Raw

def generate_sample_pcap(output_path="data/sample_traffic.pcap"):
    packets = []
    base_time = 1700000000.0  # Reference epoch timestamp

    print("Generating synthetic network capture...")

    # =========================================================================
    # 1. Benign HTTPS (TLS 1.3 / HTTPS session)
    # Flow: 192.168.1.105:51432 <-> 104.21.45.89:443 (TCP)
    # =========================================================================
    c_ip = "192.168.1.105"
    s_ip = "104.21.45.89"
    c_port = 51432
    s_port = 443
    t = base_time

    # 3-Way Handshake
    p1 = IP(src=c_ip, dst=s_ip)/TCP(sport=c_port, dport=s_port, flags="S", seq=1000, window=64240)
    p1.time = t; packets.append(p1); t += 0.020

    p2 = IP(src=s_ip, dst=c_ip)/TCP(sport=s_port, dport=c_port, flags="SA", seq=5000, ack=1001, window=65535)
    p2.time = t; packets.append(p2); t += 0.005

    p3 = IP(src=c_ip, dst=s_ip)/TCP(sport=c_port, dport=s_port, flags="A", seq=1001, ack=5001, window=64240)
    p3.time = t; packets.append(p3); t += 0.010

    # TLS Client Hello (PSH+ACK)
    client_hello = b"\x16\x03\x01\x02\x00" + b"\x01" * 512
    p4 = IP(src=c_ip, dst=s_ip)/TCP(sport=c_port, dport=s_port, flags="PA", seq=1001, ack=5001, window=64240)/Raw(load=client_hello)
    p4.time = t; packets.append(p4); t += 0.025

    # Server ACK
    p5 = IP(src=s_ip, dst=c_ip)/TCP(sport=s_port, dport=c_port, flags="A", seq=5001, ack=1001 + len(client_hello), window=65535)
    p5.time = t; packets.append(p5); t += 0.015

    # TLS Server Hello + Certificates (multi-packet response)
    server_hello_1 = b"\x16\x03\x03\x05\x00" + b"\x02" * 1400
    p6 = IP(src=s_ip, dst=c_ip)/TCP(sport=s_port, dport=c_port, flags="PA", seq=5001, ack=1001 + len(client_hello), window=65535)/Raw(load=server_hello_1)
    p6.time = t; packets.append(p6); t += 0.002

    server_hello_2 = b"\x16\x03\x03\x03\x00" + b"\x03" * 900
    p7 = IP(src=s_ip, dst=c_ip)/TCP(sport=s_port, dport=c_port, flags="PA", seq=5001 + len(server_hello_1), ack=1001 + len(client_hello), window=65535)/Raw(load=server_hello_2)
    p7.time = t; packets.append(p7); t += 0.012

    # Client ACK + Key Exchange
    client_kex = b"\x16\x03\x03\x01\x20" + b"\x04" * 280
    p8 = IP(src=c_ip, dst=s_ip)/TCP(sport=c_port, dport=s_port, flags="PA", seq=1001 + len(client_hello), ack=5001 + len(server_hello_1) + len(server_hello_2), window=64240)/Raw(load=client_kex)
    p8.time = t; packets.append(p8); t += 0.020

    # Application Data exchanges
    app_data_req = b"\x17\x03\x03\x02\x50" + b"\x05" * 600
    p9 = IP(src=c_ip, dst=s_ip)/TCP(sport=c_port, dport=s_port, flags="PA", seq=p8[TCP].seq + len(client_kex), ack=5001 + len(server_hello_1) + len(server_hello_2), window=64240)/Raw(load=app_data_req)
    p9.time = t; packets.append(p9); t += 0.035

    app_data_resp1 = b"\x17\x03\x03\x05\xb4" + b"\x06" * 1440
    p10 = IP(src=s_ip, dst=c_ip)/TCP(sport=s_port, dport=c_port, flags="A", seq=5001 + len(server_hello_1) + len(server_hello_2), ack=p9[TCP].seq + len(app_data_req), window=65535)/Raw(load=app_data_resp1)
    p10.time = t; packets.append(p10); t += 0.001

    app_data_resp2 = b"\x17\x03\x03\x04\x00" + b"\x07" * 1024
    p11 = IP(src=s_ip, dst=c_ip)/TCP(sport=s_port, dport=c_port, flags="PA", seq=p10[TCP].seq + len(app_data_resp1), ack=p9[TCP].seq + len(app_data_req), window=65535)/Raw(load=app_data_resp2)
    p11.time = t; packets.append(p11); t += 0.015

    # Connection Teardown
    p12 = IP(src=c_ip, dst=s_ip)/TCP(sport=c_port, dport=s_port, flags="FA", seq=p9[TCP].seq + len(app_data_req), ack=p11[TCP].seq + len(app_data_resp2), window=64240)
    p12.time = t; packets.append(p12); t += 0.018

    p13 = IP(src=s_ip, dst=c_ip)/TCP(sport=s_port, dport=c_port, flags="FA", seq=p11[TCP].seq + len(app_data_resp2), ack=p12[TCP].seq + 1, window=65535)
    p13.time = t; packets.append(p13); t += 0.008

    p14 = IP(src=c_ip, dst=s_ip)/TCP(sport=c_port, dport=s_port, flags="A", seq=p12[TCP].seq + 1, ack=p13[TCP].seq + 1, window=64240)
    p14.time = t; packets.append(p14); t += 0.050

    # =========================================================================
    # 2. Benign DNS (UDP query & response)
    # Flow: 192.168.1.105:54321 <-> 8.8.8.8:53 (UDP)
    # =========================================================================
    dns_q = IP(src=c_ip, dst="8.8.8.8")/UDP(sport=54321, dport=53)/Raw(load=b"\xaa\xbb\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01")
    dns_q.time = base_time + 0.005; packets.append(dns_q)

    dns_r = IP(src="8.8.8.8", dst=c_ip)/UDP(sport=53, dport=54321)/Raw(load=b"\xaa\xbb\x81\x80\x00\x01\x00\x01\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01\xc0\x0c\x00\x01\x00\x01\x00\x00\x01\x2c\x00\x04\x5d\xb8\xd8\x22")
    dns_r.time = base_time + 0.024; packets.append(dns_r)

    # =========================================================================
    # 3. PortScan Reconnaissance Attack
    # Attacker: 172.16.0.44 scanning Target: 192.168.1.50
    # Probe matching the CICIDS2017 PortScan signature (port 9, PSH flag, window 1024)
    # =========================================================================
    scan_attacker = "172.16.0.44"
    scan_target = "192.168.1.50"
    p_ps1 = IP(src=scan_attacker, dst=scan_target)/TCP(sport=48009, dport=9, flags="P", window=1024, options=[('MSS', 1460)])/Raw(load=b"\x00\x00")
    p_ps1.time = base_time + 0.5; packets.append(p_ps1)

    p_ps2 = IP(src=scan_target, dst=scan_attacker)/TCP(sport=9, dport=48009, flags="", window=0)/Raw(load=b"\x00" * 6)
    p_ps2.time = base_time + 0.500049; packets.append(p_ps2)

    # Additional scanned ports (21, 22, 23)
    for p in [21, 22, 23]:
        p_scan = IP(src=scan_attacker, dst=scan_target)/TCP(sport=48000 + p, dport=p, flags="S", window=1024)
        p_scan.time = base_time + 0.55 + p * 0.001; packets.append(p_scan)

    # =========================================================================
    # 4. DoS Attack (Denial of Service - Hulk / Slowloris signature)
    # Attacker: 10.0.0.88 targeting Target: 192.168.1.50:80
    # Microsecond burst of SYN/ACK packets with window 251, TCP timestamp options
    # =========================================================================
    dos_attacker = "10.0.0.88"
    dos_target = "192.168.1.50"
    for i in range(2):
        p_dos = IP(src=dos_attacker, dst=dos_target)/TCP(
            sport=50001,
            dport=80,
            flags="S" if i == 0 else "A",
            window=251,
            options=[('Timestamp', (10, 0))]
        )
        p_dos.time = base_time + 1.0 + i * 0.000003
        packets.append(p_dos)

    # =========================================================================
    # 5. DDoS Attack (Distributed Denial of Service Flood)
    # Attacker: 10.0.0.99 flooding Target: 192.168.1.50:80
    # Rapid packets with PSH+ACK, window 256, 6 bytes payload
    # =========================================================================
    ddos_attacker = "10.0.0.99"
    for i in range(5):
        p_ddos = IP(src=ddos_attacker, dst=dos_target)/TCP(
            sport=50002,
            dport=80,
            flags="PA",
            window=256,
            options=[('Timestamp', (10, 0))]
        )/Raw(load=b"FLOOD" * 1 + b"\x00")
        p_ddos.time = base_time + 1.5 + i * 0.4
        packets.append(p_ddos)

    # Sort all packets chronologically
    packets.sort(key=lambda p: p.time)

    # Ensure output directory exists and write capture
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wrpcap(output_path, packets)
    print(f"[SUCCESS] Wrote {len(packets)} packets across benign & attack flows to: {output_path}")

if __name__ == "__main__":
    generate_sample_pcap()
