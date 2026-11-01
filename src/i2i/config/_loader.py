from pydantic import TypeAdapter
from dataclasses import dataclass
from typing import Literal
import tomllib
import yaml
from ._config import ConfigI2I

__all__ = ["Config", "load_config"]


@dataclass
class Config:
    i2i: ConfigI2I


__type_adapter = TypeAdapter(Config)


def load_config(file: str, filetype: Literal["toml", "yaml", "json"]) -> Config:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()

    data: dict[str, object] | None = None
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        if filetype == "toml":
            raise

    if data is None:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            if filetype == "yaml" or filetype == "json":
                raise

    return __type_adapter.validate_python(data)
