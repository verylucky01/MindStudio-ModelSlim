#  -*- coding: utf-8 -*-
#  Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict

from msmodelslim.core.quant_service import DatasetLoaderInfra
from msmodelslim.utils.exception import InvalidDatasetError
from msmodelslim.utils.logging import get_logger, logger_setter
from msmodelslim.utils.security import get_valid_read_path


@dataclass
class VlmCalibSample:
    """Calibration sample for multimodal VLM."""
    text: str
    image: Optional[str] = None  # None for text-only


@logger_setter('msmodelslim.infra.vlm_dataset_loader')
class VLMDatasetLoader(DatasetLoaderInfra):
    """
    Dataset loader for multimodal vision-language models.
    
    Supports 4 types of calibration data:
    
    1. Pure text: Single json/jsonl file with text strings
       - Format: "Who are you?" or {"text": "Who are you?"}
       - Returns: [{"text": "...", "image": None}, ...]
    
    2. Pure images (default text): Image directory without json/jsonl
       - Uses default_text from config
       - Returns: [{"text": default_text, "image": "/path/to/img.jpg"}, ...]
    
    3. Pure images (custom text): Image directory + single json/jsonl
       - JSONL format: {"image": "img.jpg", "text": "What is this?"}
       - Returns: [{"text": "...", "image": "/path/to/img.jpg"}, ...]
    
    4. Mixed text and images: Directory with images + json/jsonl
       - JSONL can contain both text-only and image+text entries
       - Text-only: {"text": "Who are you?"} or just "Who are you?"
       - Image+text: {"image": "img.jpg", "text": "Describe this."}
       - Returns: mixed list of text-only and image+text samples
    
    Data structure:
        {
            'text': str,           # Text prompt/question (required)
            'image': Optional[str] # Image path (optional, None for text-only)
        }
    
    Like FileDatasetLoader, supports lab_calib short names:
        - "calibImages" -> searches in lab_calib/calibImages
        - "calib_data.jsonl" -> searches in lab_calib/calib_data.jsonl
    """
    
    def __init__(self, dataset_dir: Optional[Path] = None, default_text: str = "Describe this image in detail."):
        """
        Initialize dataset loader.
        
        Args:
            dataset_dir: Optional directory to search for relative paths (like lab_calib).
                        If None, will try to auto-detect lab_calib directory.
            default_text: Default text prompt for images without custom descriptions.
                         Defaults to "Describe this image in detail."
        """
        super().__init__()
        self.supported_image_extensions = {'.jpg', '.jpeg', '.png'}
        self.supported_data_extensions = {'.json', '.jsonl'}
        self.dataset_dir = dataset_dir
        self.default_text = default_text
    
    def get_dataset_by_name(self, dataset_name: str) -> List[VlmCalibSample]:
        """
        Load dataset by name or path.
        
        Supports 4 types:
        1. Pure text: json/jsonl file -> [{"text": "...", "image": None}, ...]
        2. Pure images (default text): image directory without json/jsonl
        3. Pure images (custom text): image directory + json/jsonl inside
        4. Mixed: directory with images + json/jsonl containing mixed entries
        
        Args:
            dataset_name: Can be:
                - Short name (e.g., "calibImages") -> searches in lab_calib/
                - Path to image directory (absolute or relative)
                - Path to json/jsonl file (absolute or relative)
                - Dataset identifier
        
        Returns:
            List of VlmCalibSample:
            - VlmCalibSample(text="Who are you?", image=None)               # text-only
            - VlmCalibSample(text="Describe this.", image="/path/img.jpg")  # image+text
        """
        dataset_path = Path(dataset_name)
        
        # Strategy (similar to FileDatasetLoader):
        # 1. If absolute path, use it directly
        # 2. If relative path exists, use it
        # 3. Otherwise, try to combine with dataset_dir (lab_calib)
        
        if dataset_path.is_absolute():
            # Case 1: Absolute path
            resolved_path = dataset_path
            get_logger().info(f"Using absolute path: {resolved_path}")
        elif dataset_path.exists():
            # Case 2: Relative path that exists from current directory
            resolved_path = dataset_path.resolve()
            get_logger().info(f"Using existing relative path: {dataset_name} -> {resolved_path}")
        else:
            # Case 3: Try to combine with dataset_dir (lab_calib)
            resolved_path = self.dataset_dir / dataset_name
            if os.path.exists(resolved_path):
                get_logger().info(f"Resolved short name: {dataset_name} -> {resolved_path}")
            else:
                # Path doesn't exist even after combining with dataset_dir
                action_hint = (
                    f"Please check if the path '{dataset_name}' is correct "
                    f"or if it exists in {self.dataset_dir}"
                )
                resolved_path = self._resolve_path_with_fallback(
                    dataset_name, dataset_path, action_hint
                )

        # Ensure path exists
        if not resolved_path.exists():
            get_logger().error(f"Dataset path not found: {resolved_path}")
            raise InvalidDatasetError(
                f"Dataset path does not exist: {resolved_path}",
                action=f"Please check if the path '{resolved_path}' is correct and exists"
            )

        # Dispatch based on type
        if resolved_path.is_dir():
            # Type 2, 3, or 4: Directory (images only, images+json/jsonl, or mixed)
            resolved_path = get_valid_read_path(str(resolved_path), is_dir=True, check_user_stat=True)
            get_logger().info(f"Loading from directory: {resolved_path}")
            return self._load_from_directory(resolved_path)
        elif resolved_path.is_file() and resolved_path.suffix.lower() in self.supported_data_extensions:
            # Type 1: Pure text json/jsonl file
            file_suffix = resolved_path.suffix.lower()
            resolved_path = get_valid_read_path(str(resolved_path), is_dir=False, check_user_stat=True)
            get_logger().info(f"Loading pure text dataset from {file_suffix} file: {resolved_path}")
            return self._load_text_from_file(resolved_path, file_suffix=file_suffix)
        else:
            # Unknown type or unsupported file format
            get_logger().error(f"Unsupported dataset type: {resolved_path}")
            raise InvalidDatasetError(
                f"Dataset path exists but is not a valid type: {resolved_path}",
                action="Please provide either a directory (images/mixed) or a json/jsonl file (text)"
            )

    def _resolve_path_with_fallback(
        self,
        dataset_name: str,
        dataset_path: Path,
        action_hint: str,
    ) -> Path:
        """
        Resolve a relative path; raise InvalidDatasetError with consistent logging and hint.
        """
        try:
            resolved_path = dataset_path.resolve()
            if resolved_path.exists():
                get_logger().info(f"Resolved relative path: {dataset_name} -> {resolved_path}")
                return resolved_path
            get_logger().error(f"Dataset path not found: {dataset_name}")
            raise InvalidDatasetError(
                f"Dataset path does not exist: {dataset_name}",
                action=action_hint,
            )
        except InvalidDatasetError:
            raise
        except Exception as e:
            get_logger().error(f"Failed to resolve path {dataset_name}: {e}")
            raise InvalidDatasetError(
                f"Failed to resolve dataset path: {dataset_name}",
                action="Please check if the path is valid and accessible",
            ) from e

    def _load_text_from_file(self, file_path: Union[Path, str], file_suffix: str = ".jsonl") -> List[VlmCalibSample]:
        """
        Load pure text dataset from json/jsonl file (Type 1).
        
        For .jsonl:
            Each line must be an independent JSON value:
            - Plain string: "Who are you?"
            - Dict with text: {"text": "Who are you?"}

        For .json:
            The entire file must be a single valid JSON value:
            - A list of strings or dicts (same formats as above)
            - Or a single string/dict
        
        Args:
            file_path: Path to json/jsonl file
        
        Returns:
            List of VlmCalibSample: [VlmCalibSample(text="...", image=None), ...]
        """
        file_path = get_valid_read_path(str(file_path))
        dataset: List[VlmCalibSample] = []

        def _push_item(item, line_num_hint: str):
            if isinstance(item, str):
                dataset.append(VlmCalibSample(text=item, image=None))
                return True
            if isinstance(item, dict) and 'text' in item:
                dataset.append(VlmCalibSample(text=item['text'], image=None))
                return True
            get_logger().warning(
                f"{line_num_hint}: Invalid format (expected string or dict with 'text'), skipping"
            )
            return False

        if file_suffix == ".jsonl":
            # Line-by-line JSONL parsing
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        ok = _push_item(item, f"Line {line_num}")
                        if not ok:
                            continue
                    except json.JSONDecodeError as e:
                        get_logger().warning(f"Line {line_num}: Invalid JSON - {e}, skipping")
                        continue
                    except Exception as e:
                        get_logger().warning(f"Line {line_num}: Error - {e}, skipping")
                        continue
        else:
            # Whole-file JSON parsing for .json
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
            except json.JSONDecodeError as e:
                raise InvalidDatasetError(
                    f"Failed to parse JSON file: {file_path}",
                    action=f"Ensure the file is a valid JSON value (list/string/dict): {e}"
                ) from e
            except Exception as e:
                raise InvalidDatasetError(
                    f"Failed to read JSON file: {file_path}",
                    action="Ensure the file is accessible and valid JSON"
                ) from e

            if isinstance(content, list):
                for idx, item in enumerate(content, 1):
                    ok = _push_item(item, f"Index {idx}")
                    if not ok:
                        continue
            else:
                ok = _push_item(content, "Root")
                if not ok:
                    get_logger().warning("Root: No valid text entry found in JSON root, skipping")

        if not dataset:
            raise InvalidDatasetError(
                f"No valid text entries found in file: {file_path}",
                action="Check json/jsonl format: use plain string or {\"text\": \"...\"}"
            )

        get_logger().info(f"Loaded {len(dataset)} pure text samples from {file_path}")
        return dataset
    
    def _load_from_directory(self, directory: Union[Path, str]) -> List[VlmCalibSample]:
        """
        Load dataset from directory (Type 2, 3, or 4).
        
        Logic:
        - If no json/jsonl: Type 2 (pure images with default text)
        - If has json/jsonl: Type 3 or 4 (images with custom text, or mixed)
        
        Args:
            directory: Path to directory
        
        Returns:
            List of VlmCalibSample: [VlmCalibSample(text="...", image="..." or None), ...]
        """
        directory = Path(get_valid_read_path(str(directory), is_dir=True, check_user_stat=True))
        
        # Find images in directory
        image_files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.supported_image_extensions
        ]
        
        # Find json/jsonl files in directory
        data_files = [
            f
            for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in self.supported_data_extensions
        ]
        
        if len(data_files) > 1:
            get_logger().warning(
                f"Found multiple json/jsonl files in {directory}, using first one: {data_files[0].name}"
            )
        
        if data_files:
            # Type 3 or 4: Has json/jsonl file
            data_file = data_files[0]
            file_suffix = data_file.suffix.lower()
            get_logger().info(f"Found data file: {data_file.name}, loading mixed/image-text dataset")
            return self._load_mixed_from_jsonl(data_file, directory, image_files, file_suffix=file_suffix)
        elif image_files:
            # Type 2: Pure images, no json/jsonl
            get_logger().info(f"No data file found, using default text for {len(image_files)} images")
            return self._load_images_with_default_text(image_files)
        else:
            raise InvalidDatasetError(
                f"No images or data files found in directory: {directory}",
                action="Ensure directory contains images (.jpg/.jpeg/.png) or a json/jsonl file"
            )
    
    def _load_images_with_default_text(self, image_files: List[Path]) -> List[VlmCalibSample]:
        """
        Load images with default text prompt (Type 2).
        
        Args:
            image_files: List of image file paths
        
        Returns:
            List of VlmCalibSample: [VlmCalibSample(text=default_text, image="/path/to/img.jpg"), ...]
        """
        get_logger().info(f"Loading {len(image_files)} images with default text: '{self.default_text}'")
        
        return [
            VlmCalibSample(text=self.default_text, image=str(img_path))
            for img_path in sorted(image_files)
        ]
    
    def _load_mixed_from_jsonl(
        self, 
        jsonl_path: Union[Path, str], 
        base_directory: Path,
        available_images: List[Path],
        file_suffix: str = ".jsonl"
    ) -> List[VlmCalibSample]:
        """
        Load mixed dataset from JSONL file (Type 3 or 4).
        
        JSONL can contain:
        - Image+text: {"image": "img.jpg", "text": "What is this?"}
        - Text-only: {"text": "Who are you?"} or just "Who are you?"
        
        For Type 3 (pure images with custom text):
        - All entries must have 'image' field
        - Image count must match available images
        
        For Type 4 (mixed):
        - Can have both image+text and text-only entries
        - Image entries must reference existing images
        
        Args:
            jsonl_path: Path to JSONL file
            base_directory: Base directory for resolving relative image paths
            available_images: List of available image files in directory
        
        Returns:
            List of VlmCalibSample: [VlmCalibSample(text="...", image="..." or None), ...]
        """
        jsonl_path = get_valid_read_path(str(jsonl_path))
        
        # Build image filename lookup for validation
        image_filenames = {img.name: img for img in available_images}
        
        dataset: List[VlmCalibSample] = []
        image_entries_count = 0
        text_only_count = 0

        def _accumulate(entry_tuple):
            nonlocal image_entries_count, text_only_count
            entry, is_image, is_text = entry_tuple
            if entry is None:
                return
            dataset.append(entry)
            image_entries_count += is_image
            text_only_count += is_text
        
        if file_suffix == ".jsonl":
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    entry_tuple = self._parse_jsonl_line(
                        line=line,
                        line_num=line_num,
                        base_directory=base_directory,
                        image_filenames=image_filenames
                    )
                    _accumulate(entry_tuple)
        else:
            # .json: whole-file JSON; support single item or list of items
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
            except json.JSONDecodeError as e:
                raise InvalidDatasetError(
                    f"Failed to parse JSON file: {jsonl_path}",
                    action=f"Ensure the file is a valid JSON value (list/dict/string): {e}"
                ) from e
            except Exception as e:
                raise InvalidDatasetError(
                    f"Failed to read JSON file: {jsonl_path}",
                    action="Ensure the file is accessible and valid JSON"
                ) from e

            def _parse_item(item, hint: str):
                # Plain string -> text-only
                if isinstance(item, str):
                    return VlmCalibSample(text=item, image=None), 0, 1

                # Dict cases
                if isinstance(item, dict):
                    has_image = 'image' in item
                    has_text = 'text' in item

                    # Text-only dict
                    if not has_image and has_text:
                        return VlmCalibSample(text=item['text'], image=None), 0, 1

                    # Image + optional text
                    if has_image:
                        image_ref = item['image']
                        text_val = item.get('text', self.default_text)

                        resolved = self._resolve_image_path(image_ref, base_directory, image_filenames, hint)
                        if resolved is None:
                            return None, 0, 0

                        return VlmCalibSample(text=text_val, image=resolved), 1, 0

                    get_logger().warning(f"{hint}: Missing both 'image' and 'text' fields, skipping")
                    return None, 0, 0

                get_logger().warning(f"{hint}: Invalid format (expected string or dict), skipping")
                return None, 0, 0

            if isinstance(content, list):
                for idx, item in enumerate(content, 1):
                    _accumulate(_parse_item(item, f"Index {idx}"))
            else:
                _accumulate(_parse_item(content, "Root"))
        
        if not dataset:
            raise InvalidDatasetError(
                f"No valid entries found in JSONL file: {jsonl_path}",
                action="Check each line: allow plain string, {\"text\": ...}, or {\"image\": ..., \"text\": ...}"
            )
        
        # Validate Type 3 vs Type 4
        if text_only_count == 0:
            # Type 3: Pure images with custom text
            get_logger().info(
                f"Loaded Type 3 dataset: {image_entries_count} images with custom text"
            )
            
            # Check if image count matches
            if image_entries_count != len(available_images):
                get_logger().warning(
                    f"Image count mismatch: JSONL has {image_entries_count} entries, "
                    f"directory has {len(available_images)} images"
                )
        else:
            # Type 4: Mixed dataset
            get_logger().info(
                f"Loaded Type 4 dataset: {image_entries_count} image+text samples, "
                f"{text_only_count} text-only samples"
            )
        
        return dataset

    def _parse_jsonl_line(
        self,
        line: str,
        line_num: int,
        base_directory: Path,
        image_filenames: Dict[str, Path],
    ) -> Tuple[Optional[VlmCalibSample], int, int]:
        """
        Parse a single jsonl line and return (entry, is_image_entry, is_text_entry).
        Returns (None, 0, 0) if the line is invalid or should be skipped.
        """
        line = line.strip()
        if not line:
            return None, 0, 0

        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            get_logger().warning(f"Line {line_num}: Invalid JSON - {e}, skipping")
            return None, 0, 0
        except Exception as e:
            get_logger().warning(f"Line {line_num}: Error - {e}, skipping")
            return None, 0, 0

        # Case 1: plain string -> text-only
        if isinstance(item, str):
            return VlmCalibSample(text=item, image=None), 0, 1

        # Case 2: dict
        if isinstance(item, dict):
            if 'image' in item:
                resolved = self._resolve_image_path(
                    item['image'],
                    base_directory,
                    image_filenames,
                    hint=f"Line {line_num}"
                )
                if resolved is None:
                    return None, 0, 0
                text = item.get('text', self.default_text)
                return VlmCalibSample(text=text, image=resolved), 1, 0

            if 'text' in item:
                return VlmCalibSample(text=item['text'], image=None), 0, 1

            get_logger().warning(
                f"Line {line_num}: Missing both 'image' and 'text' fields, skipping"
            )
            return None, 0, 0

        get_logger().warning(
            f"Line {line_num}: Invalid format (expected string or dict), skipping"
        )
        return None, 0, 0

    def _resolve_image_path(
        self,
        image_ref: str,
        base_directory: Path,
        image_filenames: Dict[str, Path],
        hint: str
    ) -> Optional[str]:
        """
        Resolve and validate image path for mixed json/jsonl entries.
        Returns validated path (str) or None if invalid.
        """
        image_path = Path(image_ref)
        if not image_path.is_absolute():
            if image_path.name in image_filenames:
                resolved_image = image_filenames[image_path.name]
            else:
                resolved_image = base_directory / image_ref
                if not resolved_image.exists():
                    get_logger().warning(f"{hint}: Image '{image_ref}' not found in directory, skipping")
                    return None
        else:
            resolved_image = image_path
            if not resolved_image.exists():
                get_logger().warning(f"{hint}: Image path '{image_ref}' does not exist, skipping")
                return None

        try:
            return get_valid_read_path(str(resolved_image))
        except Exception as e:
            get_logger().warning(f"{hint}: Failed to validate image path '{resolved_image}': {e}, skipping")
            return None