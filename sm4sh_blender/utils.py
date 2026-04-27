import struct
from typing import Optional


def get_enum_value(obj, name: str, enum, default=None):
    # Support custom properties like obj.custom_prop or obj["custom_prop"].
    if value := getattr(obj, name, None) or obj.get(name):
        return getattr(enum, value, default)

    return default


def float32_from_bits(bits: int) -> float:
    return struct.unpack("@f", struct.pack("@I", bits))[0]


def parse_int(name: str, base=10) -> Optional[int]:
    value = None
    try:
        value = int(name, base)
    except:
        value = None

    return value


def extract_name(name: str, separator: str) -> str:
    return name.split(separator)[0] if separator in name else name
