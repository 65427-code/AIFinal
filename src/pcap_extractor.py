"""
AI-Based Encrypted Traffic Threat Detection
PCAP Flow Feature Extractor — Rewritten v2.0

Parses .pcap/.pcapng files into bidirectional network flows using proper
connection semantics and computes CICFlowMeter-compatible statistical
features without payload decryption.

Key improvements over v1:
- TCP FIN/RST flow termination
- Configurable idle and active timeouts
- Separate flows on 5-tuple reuse after termination
- Proper active/idle period segmentation
- Streaming packet reader (no hardcoded limit)
- Truncation reporting
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Any
from collections import defaultdict

from scapy.all import PcapReader, IP, IPv6, TCP, UDP

from src.features.schema import CANONICAL_FEATURES
from src.config import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_ACTIVE_TIMEOUT,
    DEFAULT_UDP_TIMEOUT,
    ACTIVE_IDLE_THRESHOLD,
    DEFAULT_MAX_PACKETS,
)


# ─── Flow State Machine ────────────────────────────────────────────────────

class FlowState:
    """Tracks the state of a single bidirectional network flow."""
    
    __slots__ = [
        'fwd_src_ip', 'fwd_dst_ip', 'fwd_src_port', 'fwd_dst_port',
        'protocol', 'is_tcp', 'start_time', 'last_time',
        'fwd_pkt_lens', 'bwd_pkt_lens', 'fwd_pkt_times', 'bwd_pkt_times',
        'all_pkt_lens', 'all_pkt_times',
        'fwd_header_lens', 'bwd_header_lens',
        'fwd_tcp_flags', 'bwd_tcp_flags', 'all_tcp_flags',
        'fwd_win_sizes', 'bwd_win_sizes',
        'fwd_payload_lens', 'bwd_payload_lens',
        'fin_count', 'rst_seen', 'terminated',
        'total_fwd_pkts', 'total_bwd_pkts',
    ]
    
    def __init__(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                 protocol: int, is_tcp: bool, timestamp: float):
        self.fwd_src_ip = src_ip
        self.fwd_dst_ip = dst_ip
        self.fwd_src_port = src_port
        self.fwd_dst_port = dst_port
        self.protocol = protocol
        self.is_tcp = is_tcp
        self.start_time = timestamp
        self.last_time = timestamp
        
        self.fwd_pkt_lens: List[int] = []
        self.bwd_pkt_lens: List[int] = []
        self.fwd_pkt_times: List[float] = []
        self.bwd_pkt_times: List[float] = []
        self.all_pkt_lens: List[int] = []
        self.all_pkt_times: List[float] = []
        
        self.fwd_header_lens: List[int] = []
        self.bwd_header_lens: List[int] = []
        
        self.fwd_tcp_flags: List[dict] = []
        self.bwd_tcp_flags: List[dict] = []
        self.all_tcp_flags: List[dict] = []
        
        self.fwd_win_sizes: List[int] = []
        self.bwd_win_sizes: List[int] = []
        
        self.fwd_payload_lens: List[int] = []
        self.bwd_payload_lens: List[int] = []
        
        self.fin_count = 0
        self.rst_seen = False
        self.terminated = False
        
        self.total_fwd_pkts = 0
        self.total_bwd_pkts = 0
    
    def add_packet(self, pkt_len: int, payload_len: int, header_len: int,
                   timestamp: float, is_forward: bool, tcp_flags: dict,
                   win_size: int):
        """Add a packet to this flow."""
        self.last_time = timestamp
        self.all_pkt_lens.append(payload_len)
        self.all_pkt_times.append(timestamp)
        self.all_tcp_flags.append(tcp_flags)
        
        if is_forward:
            self.total_fwd_pkts += 1
            self.fwd_pkt_lens.append(payload_len)
            self.fwd_pkt_times.append(timestamp)
            self.fwd_header_lens.append(header_len)
            self.fwd_tcp_flags.append(tcp_flags)
            self.fwd_win_sizes.append(win_size)
            self.fwd_payload_lens.append(payload_len)
        else:
            self.total_bwd_pkts += 1
            self.bwd_pkt_lens.append(payload_len)
            self.bwd_pkt_times.append(timestamp)
            self.bwd_header_lens.append(header_len)
            self.bwd_tcp_flags.append(tcp_flags)
            self.bwd_win_sizes.append(win_size)
            self.bwd_payload_lens.append(payload_len)
        
        # Track FIN/RST for flow termination
        if tcp_flags.get("FIN", 0):
            self.fin_count += 1
        if tcp_flags.get("RST", 0):
            self.rst_seen = True
    
    @property
    def should_terminate(self) -> bool:
        """Check if flow should be terminated based on TCP flags."""
        if self.rst_seen:
            return True
        # Both sides sent FIN
        if self.fin_count >= 2:
            return True
        return False

    @property
    def fwd_pkt_count(self) -> int:
        return self.total_fwd_pkts

    @property
    def bwd_pkt_count(self) -> int:
        return self.total_bwd_pkts

    @property
    def src_ip(self) -> str:
        return self.fwd_src_ip

    @property
    def dst_ip(self) -> str:
        return self.fwd_dst_ip


# ─── Flow Table ─────────────────────────────────────────────────────────────

class FlowTable:
    """
    Manages active network flows with proper lifecycle semantics.
    
    Handles:
    - Bidirectional 5-tuple keying
    - TCP FIN/RST termination
    - Idle timeout expiration
    - Active/flow timeout expiration
    - Generation counter for 5-tuple reuse
    """
    
    def __init__(
        self,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        active_timeout: float = DEFAULT_ACTIVE_TIMEOUT,
        udp_timeout: float = DEFAULT_UDP_TIMEOUT,
        flow_timeout: Optional[float] = None,
    ):
        self.idle_timeout = flow_timeout if flow_timeout is not None else idle_timeout
        self.active_timeout = active_timeout
        self.udp_timeout = udp_timeout
        
        # Active flows: canonical_key → FlowState
        self.active_flows: Dict[tuple, FlowState] = {}
        # Completed flows
        self.completed_flows: List[FlowState] = []
        # Generation counter per 5-tuple for reuse
        self._generation: Dict[tuple, int] = defaultdict(int)

    @property
    def finished_flows(self) -> List[FlowState]:
        return self.completed_flows
    
    def _canonical_key(self, src_ip: str, dst_ip: str, src_port: int,
                       dst_port: int, protocol: int) -> tuple:
        """Create canonical bidirectional 5-tuple key."""
        ep1 = (src_ip, src_port)
        ep2 = (dst_ip, dst_port)
        if ep1 <= ep2:
            base = (ep1[0], ep1[1], ep2[0], ep2[1], protocol)
        else:
            base = (ep2[0], ep2[1], ep1[0], ep1[1], protocol)
        return base
    
    def _flow_key(self, base_key: tuple) -> tuple:
        """Add generation counter to base key for uniqueness."""
        gen = self._generation[base_key]
        return base_key + (gen,)
    
    def expire_flows(self, current_time: float):
        """Expire flows that have exceeded timeout thresholds."""
        expired_keys = []
        for key, flow in self.active_flows.items():
            elapsed = current_time - flow.last_time
            timeout = self.udp_timeout if not flow.is_tcp else self.idle_timeout
            
            if elapsed > timeout:
                expired_keys.append(key)
            elif (current_time - flow.start_time) > self.active_timeout:
                expired_keys.append(key)
        
        for key in expired_keys:
            flow = self.active_flows.pop(key)
            self.completed_flows.append(flow)
    
    def process_packet(
        self,
        src_ip: str, dst_ip: str, src_port: int, dst_port: int,
        protocol: int, is_tcp: bool,
        pkt_len: int, payload_len: int, header_len: int,
        timestamp: float, tcp_flags: dict, win_size: int,
    ):
        """Process a single packet and assign it to the appropriate flow."""
        base_key = self._canonical_key(src_ip, dst_ip, src_port, dst_port, protocol)
        flow_key = self._flow_key(base_key)
        
        if flow_key in self.active_flows:
            flow = self.active_flows[flow_key]
            
            # Check idle timeout
            elapsed = timestamp - flow.last_time
            timeout = self.udp_timeout if not is_tcp else self.idle_timeout
            
            if elapsed > timeout or (timestamp - flow.start_time) > self.active_timeout:
                # Expire this flow, start a new one
                self.completed_flows.append(flow)
                del self.active_flows[flow_key]
                self._generation[base_key] += 1
                flow_key = self._flow_key(base_key)
                # Create new flow below
                flow = None
            elif flow.terminated:
                # Flow was terminated (FIN/RST), start new generation
                self.completed_flows.append(flow)
                del self.active_flows[flow_key]
                self._generation[base_key] += 1
                flow_key = self._flow_key(base_key)
                flow = None
            else:
                # Determine direction
                is_forward = (src_ip == flow.fwd_src_ip and src_port == flow.fwd_src_port)
                flow.add_packet(pkt_len, payload_len, header_len, timestamp,
                               is_forward, tcp_flags, win_size)
                
                # Check if flow should terminate after this packet
                if flow.should_terminate:
                    flow.terminated = True
                    self.completed_flows.append(flow)
                    del self.active_flows[flow_key]
                    self._generation[base_key] += 1
                return
        else:
            flow = None
        
        # Create new flow
        new_flow = FlowState(
            src_ip=src_ip, dst_ip=dst_ip,
            src_port=src_port, dst_port=dst_port,
            protocol=protocol, is_tcp=is_tcp,
            timestamp=timestamp,
        )
        new_flow.add_packet(pkt_len, payload_len, header_len, timestamp,
                           True, tcp_flags, win_size)  # First packet is always forward
        
        if new_flow.should_terminate:
            new_flow.terminated = True
            self.completed_flows.append(new_flow)
            self._generation[base_key] += 1
        else:
            self.active_flows[flow_key] = new_flow
    
    def flush_all(self):
        """Move all remaining active flows to completed."""
        for key, flow in self.active_flows.items():
            self.completed_flows.append(flow)
        self.active_flows.clear()
    
    def get_all_flows(self) -> List[FlowState]:
        """Get all completed flows (call flush_all first)."""
        return self.completed_flows


# ─── Feature Calculation ────────────────────────────────────────────────────

def _calc_iat(times: List[float]) -> Tuple[float, float, float, float, float]:
    """
    Calculate inter-arrival time statistics in microseconds.
    Returns: (total, mean, std, max, min)
    """
    if len(times) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = [(times[i] - times[i - 1]) * 1e6 for i in range(1, len(times))]
    return (
        sum(diffs),
        float(np.mean(diffs)),
        float(np.std(diffs)) if len(diffs) > 1 else 0.0,
        float(np.max(diffs)),
        float(np.min(diffs)),
    )


class ActiveIdleResult(tuple):
    """Container supporting both tuple unpacking (active_times, idle_times) and dict lookup for summary metrics."""
    def __new__(cls, active_times, idle_times):
        return super().__new__(cls, (active_times, idle_times))
    
    @property
    def active_times(self) -> list:
        return self[0]
        
    @property
    def idle_times(self) -> list:
        return self[1]
        
    def __getitem__(self, key):
        if isinstance(key, str):
            active_times = self[0]
            idle_times = self[1]
            if key == 'Active Mean':
                return float(np.mean(active_times)) if active_times else 0.0
            elif key == 'Active Std':
                return float(np.std(active_times)) if len(active_times) > 1 else 0.0
            elif key == 'Active Max':
                return float(np.max(active_times)) if active_times else 0.0
            elif key == 'Active Min':
                return float(np.min(active_times)) if active_times else 0.0
            elif key == 'Idle Mean':
                return float(np.mean(idle_times)) if idle_times else 0.0
            elif key == 'Idle Std':
                return float(np.std(idle_times)) if len(idle_times) > 1 else 0.0
            elif key == 'Idle Max':
                return float(np.max(idle_times)) if idle_times else 0.0
            elif key == 'Idle Min':
                return float(np.min(idle_times)) if idle_times else 0.0
            raise KeyError(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        return key in ('Active Mean', 'Active Std', 'Active Max', 'Active Min',
                       'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min')


def _calc_active_idle(times: List[float], threshold_sec: Any = ACTIVE_IDLE_THRESHOLD):
    """
    Segment flow into active and idle periods.
    
    An idle period occurs when the inter-packet gap exceeds threshold_sec.
    An active period is a contiguous burst of packets with gaps <= threshold_sec.
    
    Returns:
        ActiveIdleResult: tuple of (active_times, idle_times) that also supports dict access.
    """
    if isinstance(threshold_sec, (list, tuple)):
        all_times = sorted(list(times) + list(threshold_sec))
        thresh = ACTIVE_IDLE_THRESHOLD
    else:
        all_times = sorted(list(times))
        thresh = float(threshold_sec)
        
    if len(all_times) < 2:
        return ActiveIdleResult([], [])
    
    active_times = []
    idle_times = []
    
    active_start = all_times[0]
    
    for i in range(1, len(all_times)):
        gap = all_times[i] - all_times[i - 1]
        
        if gap > thresh:
            # End of active period, start of idle period
            active_duration = (all_times[i - 1] - active_start) * 1e6
            if active_duration > 0:
                active_times.append(active_duration)
            
            idle_duration = gap * 1e6
            idle_times.append(idle_duration)
            
            # New active period starts
            active_start = all_times[i]
    
    # Final active period
    final_active = (all_times[-1] - active_start) * 1e6
    if final_active > 0:
        active_times.append(final_active)
    
    return ActiveIdleResult(active_times, idle_times)


def compute_flow_features(flow: FlowState) -> Dict[str, float]:
    """
    Compute all 65 canonical CICFlowMeter-compatible features from a FlowState.
    """
    tot_fwd = flow.total_fwd_pkts
    tot_bwd = flow.total_bwd_pkts
    tot_pkts = tot_fwd + tot_bwd
    
    # Duration in microseconds
    duration_sec = flow.last_time - flow.start_time
    duration_us = max(0.0, duration_sec * 1e6)
    
    # Packet length statistics (payload)
    fwd_lens = flow.fwd_pkt_lens if flow.fwd_pkt_lens else [0]
    bwd_lens = flow.bwd_pkt_lens if flow.bwd_pkt_lens else [0]
    all_lens = flow.all_pkt_lens if flow.all_pkt_lens else [0]
    
    tot_len_fwd = sum(flow.fwd_pkt_lens)
    tot_len_bwd = sum(flow.bwd_pkt_lens)
    
    fwd_max = float(np.max(fwd_lens)) if flow.fwd_pkt_lens else 0.0
    fwd_min = float(np.min(fwd_lens)) if flow.fwd_pkt_lens else 0.0
    fwd_mean = float(np.mean(fwd_lens)) if flow.fwd_pkt_lens else 0.0
    fwd_std = float(np.std(fwd_lens)) if len(flow.fwd_pkt_lens) > 1 else 0.0
    
    bwd_max = float(np.max(bwd_lens)) if flow.bwd_pkt_lens else 0.0
    bwd_min = float(np.min(bwd_lens)) if flow.bwd_pkt_lens else 0.0
    bwd_mean = float(np.mean(bwd_lens)) if flow.bwd_pkt_lens else 0.0
    bwd_std = float(np.std(bwd_lens)) if len(flow.bwd_pkt_lens) > 1 else 0.0
    
    pkt_max = float(np.max(all_lens))
    pkt_min = float(np.min(all_lens))
    pkt_mean = float(np.mean(all_lens))
    pkt_std = float(np.std(all_lens)) if len(all_lens) > 1 else 0.0
    pkt_var = float(np.var(all_lens)) if len(all_lens) > 1 else 0.0
    
    # Rates
    dur_div = duration_sec if duration_sec > 0 else 1e-6
    flow_bytes_s = (tot_len_fwd + tot_len_bwd) / dur_div
    flow_pkts_s = tot_pkts / dur_div
    fwd_pkts_s = tot_fwd / dur_div
    bwd_pkts_s = tot_bwd / dur_div
    
    # Inter-arrival times
    _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _calc_iat(flow.all_pkt_times)
    fwd_iat_tot, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _calc_iat(flow.fwd_pkt_times)
    bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _calc_iat(flow.bwd_pkt_times)
    
    # TCP Flags (summed across all packets)
    fin_cnt = sum(f.get("FIN", 0) for f in flow.all_tcp_flags)
    syn_cnt = sum(f.get("SYN", 0) for f in flow.all_tcp_flags)
    psh_cnt = sum(f.get("PSH", 0) for f in flow.all_tcp_flags)
    ack_cnt = sum(f.get("ACK", 0) for f in flow.all_tcp_flags)
    urg_cnt = sum(f.get("URG", 0) for f in flow.all_tcp_flags)
    fwd_psh_flags = sum(f.get("PSH", 0) for f in flow.fwd_tcp_flags)
    
    # Header lengths (transport header)
    fwd_hdr_len = sum(flow.fwd_header_lens)
    bwd_hdr_len = sum(flow.bwd_header_lens)
    
    # Initial window sizes
    init_win_fwd = flow.fwd_win_sizes[0] if flow.fwd_win_sizes else -1
    init_win_bwd = flow.bwd_win_sizes[0] if flow.bwd_win_sizes else -1
    
    # act_data_pkt_fwd: forward packets with payload > 0
    act_data_fwd = sum(1 for p in flow.fwd_payload_lens if p > 0)
    
    # min_seg_size_forward: minimum transport header length in forward direction
    min_seg_fwd = min(flow.fwd_header_lens) if flow.fwd_header_lens else 20
    
    # Down/Up ratio
    down_up_ratio = (tot_bwd / tot_fwd) if tot_fwd > 0 else 0.0
    
    # Average packet size
    avg_pkt_size = sum(all_lens) / tot_pkts if tot_pkts > 0 else 0.0
    
    # Active/Idle segmentation
    active_times, idle_times = _calc_active_idle(flow.all_pkt_times)
    
    if active_times:
        active_mean = float(np.mean(active_times))
        active_std = float(np.std(active_times)) if len(active_times) > 1 else 0.0
        active_max = float(np.max(active_times))
        active_min = float(np.min(active_times))
    else:
        # Single burst — entire flow is one active period
        active_mean = duration_us
        active_std = 0.0
        active_max = duration_us
        active_min = duration_us
    
    if idle_times:
        idle_mean = float(np.mean(idle_times))
        idle_std = float(np.std(idle_times)) if len(idle_times) > 1 else 0.0
        idle_max = float(np.max(idle_times))
        idle_min = float(np.min(idle_times))
    else:
        idle_mean = idle_std = idle_max = idle_min = 0.0
    
    return {
        "Destination Port": flow.fwd_dst_port,
        "Flow Duration": duration_us,
        "Total Fwd Packets": tot_fwd,
        "Total Backward Packets": tot_bwd,
        "Total Length of Fwd Packets": tot_len_fwd,
        "Total Length of Bwd Packets": tot_len_bwd,
        "Fwd Packet Length Max": fwd_max,
        "Fwd Packet Length Min": fwd_min,
        "Fwd Packet Length Mean": fwd_mean,
        "Fwd Packet Length Std": fwd_std,
        "Bwd Packet Length Max": bwd_max,
        "Bwd Packet Length Min": bwd_min,
        "Bwd Packet Length Mean": bwd_mean,
        "Bwd Packet Length Std": bwd_std,
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,
        "Fwd IAT Total": fwd_iat_tot,
        "Fwd IAT Mean": fwd_iat_mean,
        "Fwd IAT Std": fwd_iat_std,
        "Fwd IAT Max": fwd_iat_max,
        "Fwd IAT Min": fwd_iat_min,
        "Bwd IAT Total": bwd_iat_tot,
        "Bwd IAT Mean": bwd_iat_mean,
        "Bwd IAT Std": bwd_iat_std,
        "Bwd IAT Max": bwd_iat_max,
        "Bwd IAT Min": bwd_iat_min,
        "Fwd PSH Flags": fwd_psh_flags,
        "Fwd Header Length": fwd_hdr_len,
        "Bwd Header Length": bwd_hdr_len,
        "Fwd Packets/s": fwd_pkts_s,
        "Bwd Packets/s": bwd_pkts_s,
        "Min Packet Length": pkt_min,
        "Max Packet Length": pkt_max,
        "Packet Length Mean": pkt_mean,
        "Packet Length Std": pkt_std,
        "Packet Length Variance": pkt_var,
        "FIN Flag Count": fin_cnt,
        "SYN Flag Count": syn_cnt,
        "PSH Flag Count": psh_cnt,
        "ACK Flag Count": ack_cnt,
        "URG Flag Count": urg_cnt,
        "Down/Up Ratio": down_up_ratio,
        "Average Packet Size": avg_pkt_size,
        "Avg Fwd Segment Size": fwd_mean,
        "Avg Bwd Segment Size": bwd_mean,
        "Subflow Fwd Packets": tot_fwd,
        "Subflow Fwd Bytes": tot_len_fwd,
        "Subflow Bwd Packets": tot_bwd,
        "Subflow Bwd Bytes": tot_len_bwd,
        "Init_Win_bytes_forward": init_win_fwd,
        "Init_Win_bytes_backward": init_win_bwd,
        "act_data_pkt_fwd": act_data_fwd,
        "min_seg_size_forward": min_seg_fwd,
        "Active Mean": active_mean,
        "Active Std": active_std,
        "Active Max": active_max,
        "Active Min": active_min,
        "Idle Mean": idle_mean,
        "Idle Std": idle_std,
        "Idle Max": idle_max,
        "Idle Min": idle_min,
    }


# ─── Main Extraction Function ──────────────────────────────────────────────

def extract_flows_from_pcap(
    pcap_source,
    max_packets: Optional[int] = DEFAULT_MAX_PACKETS,
    idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
    active_timeout: float = DEFAULT_ACTIVE_TIMEOUT,
    report_progress: bool = False,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Extract bidirectional network flows and compute features from a PCAP file.
    
    Args:
        pcap_source: File path (str) or file-like stream (e.g., Streamlit UploadedFile)
        max_packets: Maximum packets to process (None = unlimited)
        idle_timeout: Seconds of inactivity before a flow is considered expired
        active_timeout: Maximum duration of a single flow in seconds
        report_progress: If True, print progress every 100K packets
    
    Returns:
        Tuple of:
          - DataFrame with flow features + metadata columns
          - List of flow metadata dicts
    """
    # Handle file-like objects (Streamlit uploads)
    temp_file_created = False
    if hasattr(pcap_source, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pcap") as tmp:
            pcap_source.seek(0)
            tmp.write(pcap_source.read())
            pcap_path = tmp.name
            temp_file_created = True
    else:
        pcap_path = pcap_source
    
    flow_table = FlowTable(
        idle_timeout=idle_timeout,
        active_timeout=active_timeout,
    )
    
    pkt_count = 0
    truncated = False
    total_capture_packets = None
    
    try:
        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                pkt_count += 1
                
                if max_packets is not None and pkt_count > max_packets:
                    truncated = True
                    break
                
                if report_progress and pkt_count % 100000 == 0:
                    print(f"  Processed {pkt_count:,} packets, "
                          f"{len(flow_table.active_flows)} active flows, "
                          f"{len(flow_table.completed_flows)} completed flows")
                
                # Only process IP packets with TCP or UDP
                if not (IP in pkt or IPv6 in pkt):
                    continue
                
                ip_layer = pkt[IP] if IP in pkt else pkt[IPv6]
                src_ip = ip_layer.src
                dst_ip = ip_layer.dst
                proto = ip_layer.proto if IP in pkt else ip_layer.nh
                
                if TCP in pkt:
                    transport = pkt[TCP]
                    src_port = transport.sport
                    dst_port = transport.dport
                    is_tcp = True
                    
                    # Calculate header lengths
                    ip_hdr_len = ip_layer.ihl * 4 if IP in pkt else 40
                    trans_hdr_len = transport.dataofs * 4 if transport.dataofs else 20
                    header_len = trans_hdr_len  # Store transport header length
                    payload_len = max(0, len(pkt) - ip_hdr_len - trans_hdr_len)
                    
                    # TCP flags
                    flags_val = int(transport.flags)
                    tcp_flags = {
                        "FIN": 1 if flags_val & 0x01 else 0,
                        "SYN": 1 if flags_val & 0x02 else 0,
                        "RST": 1 if flags_val & 0x04 else 0,
                        "PSH": 1 if flags_val & 0x08 else 0,
                        "ACK": 1 if flags_val & 0x10 else 0,
                        "URG": 1 if flags_val & 0x20 else 0,
                    }
                    win_size = transport.window
                    
                elif UDP in pkt:
                    transport = pkt[UDP]
                    src_port = transport.sport
                    dst_port = transport.dport
                    is_tcp = False
                    
                    ip_hdr_len = ip_layer.ihl * 4 if IP in pkt else 40
                    trans_hdr_len = 8
                    header_len = trans_hdr_len
                    payload_len = max(0, len(pkt) - ip_hdr_len - 8)
                    
                    tcp_flags = {"FIN": 0, "SYN": 0, "RST": 0, "PSH": 0, "ACK": 0, "URG": 0}
                    win_size = -1
                else:
                    continue
                
                ts = float(pkt.time)
                pkt_len = len(pkt)
                
                # Periodically expire idle flows (every 10K packets)
                if pkt_count % 10000 == 0:
                    flow_table.expire_flows(ts)
                
                flow_table.process_packet(
                    src_ip=src_ip, dst_ip=dst_ip,
                    src_port=src_port, dst_port=dst_port,
                    protocol=proto, is_tcp=is_tcp,
                    pkt_len=pkt_len, payload_len=payload_len,
                    header_len=header_len, timestamp=ts,
                    tcp_flags=tcp_flags, win_size=win_size,
                )
    
    finally:
        if temp_file_created and os.path.exists(pcap_path):
            try:
                os.remove(pcap_path)
            except Exception:
                pass
    
    # Flush remaining active flows
    flow_table.flush_all()
    all_flows = flow_table.get_all_flows()
    
    if not all_flows:
        return pd.DataFrame(), []
    
    # Compute features for each flow
    flow_rows = []
    metadata_rows = []
    
    for flow in all_flows:
        features = compute_flow_features(flow)
        flow_rows.append(features)
        
        total_pkts = flow.total_fwd_pkts + flow.total_bwd_pkts
        total_bytes = sum(flow.all_pkt_lens)
        
        meta = {
            "Source_IP": flow.fwd_src_ip,
            "Destination_IP": flow.fwd_dst_ip,
            "Source_Port": flow.fwd_src_port,
            "Destination_Port": flow.fwd_dst_port,
            "Protocol": "TCP" if flow.is_tcp else ("UDP" if flow.protocol == 17 else str(flow.protocol)),
            "Packets": total_pkts,
            "Bytes": total_bytes,
        }
        metadata_rows.append(meta)
    
    features_df = pd.DataFrame(flow_rows)
    meta_df = pd.DataFrame(metadata_rows)
    combined_df = pd.concat([meta_df, features_df], axis=1)
    
    # Add truncation info to metadata
    extraction_info = {
        "packets_processed": pkt_count - (1 if truncated else 0),
        "flows_extracted": len(all_flows),
        "was_truncated": truncated,
        "max_packets_setting": max_packets,
    }
    
    # Store extraction info as attributes on the dataframe
    combined_df.attrs["extraction_info"] = extraction_info
    
    return combined_df, metadata_rows


# ─── CLI Interface ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract flows from PCAP files")
    parser.add_argument("pcap", help="Path to PCAP/PCAPNG file")
    parser.add_argument("--max-packets", type=int, default=None,
                       help="Maximum packets to process (default: unlimited)")
    parser.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT,
                       help=f"Idle timeout in seconds (default: {DEFAULT_IDLE_TIMEOUT})")
    parser.add_argument("--active-timeout", type=float, default=DEFAULT_ACTIVE_TIMEOUT,
                       help=f"Active timeout in seconds (default: {DEFAULT_ACTIVE_TIMEOUT})")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output CSV file path")
    
    args = parser.parse_args()
    
    print(f"Extracting flows from: {args.pcap}")
    print(f"  Idle timeout: {args.idle_timeout}s")
    print(f"  Active timeout: {args.active_timeout}s")
    print(f"  Max packets: {'unlimited' if args.max_packets is None else args.max_packets}")
    
    df, meta = extract_flows_from_pcap(
        args.pcap,
        max_packets=args.max_packets,
        idle_timeout=args.idle_timeout,
        active_timeout=args.active_timeout,
        report_progress=True,
    )
    
    info = df.attrs.get("extraction_info", {})
    print(f"\nResults:")
    print(f"  Packets processed: {info.get('packets_processed', 'N/A'):,}")
    print(f"  Flows extracted: {info.get('flows_extracted', len(df)):,}")
    print(f"  Truncated: {'Yes' if info.get('was_truncated') else 'No'}")
    
    if not df.empty:
        print(f"\nSample flows:")
        display_cols = ["Source_IP", "Destination_IP", "Destination_Port", "Protocol", "Packets"]
        print(df[[c for c in display_cols if c in df.columns]].head(10).to_string())
        
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved to: {args.output}")
