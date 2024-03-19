# Authors: Vasisht Duddu, Anudeep Das, Nora Khayata, Hossein Yalame, Thomas Schneider, N Asokan
# Copyright 2024 Secure Systems Group, University of Waterloo & Aalto University, https://crysp.uwaterloo.ca/research/SSG/
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Optional, Any
import logging


def create_dir_if_doesnt_exist(path_to_dir: Path, log: logging.Logger) -> Optional[Path]:
    """Create directory using provided path.
    No explicit checks after creation because pathlib verifies it itself.
    Check pathlib source if errors happen.

    Args:
        path_to_dir (Path): Directory to be created
        log (logging.Logger): Logging facility

    Returns:
        Optional[Path]: Maybe Path to the created directory
    """
    resolved_path: Path = path_to_dir.resolve()
    already_exists = False

    if not resolved_path.exists():
        log.info("{} does not exist. Creating...")

        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.error("Error occurred while creating directory {}. Exception: {}".format(resolved_path, e))
            return None

        log.info("{} created.".format(resolved_path))
    else:
        log.info("{} already exists.".format(resolved_path))
        already_exists = True

    return resolved_path, already_exists
