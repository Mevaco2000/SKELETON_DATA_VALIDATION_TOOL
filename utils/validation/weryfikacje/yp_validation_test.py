"""Lightweight container for validation experiments and ad-hoc checks."""

import ast
import os
from collections.abc import Mapping
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from ...datasets.yolo_pose_dataset import YOLOPoseDataset


class YPValidation_Test:
    """Store and summarize results from manual validation tests.

    This class is intentionally small. It provides a stable place for
    temporary validation logic without coupling it to YPImageValidation or
    YPSetValidation.
    """

    def __init__(self, name: str = "YPValidation_Test", metadata: Optional[Dict[str, Any]] = None) -> None:
        self.name = name
        self.metadata = dict(metadata or {})
        self._results: List[Dict[str, Any]] = []

    @property
    def results(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of collected results."""
        return list(self._results)

    def add_result(self, label: str, value: Any, **extra: Any) -> None:
        """Append a single validation result entry."""
        entry = {"label": label, "value": value}
        if extra:
            entry.update(extra)
        self._results.append(entry)

    def extend_results(self, entries: List[Dict[str, Any]]) -> None:
        """Append multiple precomputed result entries."""
        self._results.extend(dict(entry) for entry in entries)

    def clear(self) -> None:
        """Remove all collected results."""
        self._results.clear()

    @staticmethod
    def parse_results_txt_value(raw_value: str) -> Any:
        """Convert a raw results.txt field to a Python object when possible."""
        value = raw_value.strip()
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    @classmethod
    def parse_results_txt_line(cls, line: str) -> Dict[str, Any]:
        """Parse a single line from a validation results.txt file."""
        stripped = line.strip()
        if not stripped:
            raise ValueError("Empty results.txt line cannot be parsed")

        parts = [part.strip() for part in stripped.split(" | ") if part.strip()]
        parsed: Dict[str, Any] = {}

        first_part = parts[0]
        if first_part.startswith("{") and first_part.endswith("}"):
            parsed.update(cls.parse_results_txt_value(first_part))
            parts = parts[1:]

        for part in parts:
            if ": " not in part:
                continue
            key, raw_value = part.split(": ", 1)
            parsed[key.strip()] = cls.parse_results_txt_value(raw_value)

        if not parsed:
            raise ValueError(f"Could not parse results.txt line: {line[:120]}")

        return parsed

    @classmethod
    def parse_results_txt(cls, results_txt_path: str) -> List[Dict[str, Any]]:
        """Parse a validation results.txt file into structured records."""
        parsed_entries: List[Dict[str, Any]] = []

        with open(results_txt_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    entry = cls.parse_results_txt_line(stripped)
                except ValueError as exc:
                    raise ValueError(
                        f"Failed to parse {results_txt_path} at line {line_number}: {exc}"
                    ) from exc

                entry["line_number"] = line_number
                parsed_entries.append(entry)

        return parsed_entries

    @classmethod
    def parse_detected_inconsistencies_txt(cls, report_path: str) -> List[Dict[str, Any]]:
        """Parse the plain-text common inconsistencies report into structured records."""
        parsed_entries: List[Dict[str, Any]] = []

        with open(report_path, "r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped or not stripped.startswith("[") or "image_path:" not in stripped:
                    continue

                parts = [part.strip() for part in stripped.split(" | ") if part.strip()]
                if not parts:
                    continue

                first_part = parts[0]
                _, _, first_value = first_part.partition(": ")
                if not first_value:
                    continue

                entry: Dict[str, Any] = {
                    "image_path": first_value.strip(),
                    "line_number": line_number,
                }

                for part in parts[1:]:
                    if ": " not in part:
                        continue
                    key, raw_value = part.split(": ", 1)
                    entry[key.strip()] = cls.parse_results_txt_value(raw_value)

                parsed_entries.append(entry)

        return parsed_entries

    @classmethod
    def parse_candidate_results_file(cls, candidate_path: str) -> List[Dict[str, Any]]:
        """Parse a candidate comparison file, auto-detecting supported text formats."""
        with open(candidate_path, "r", encoding="utf-8") as file:
            first_nonempty_line = ""
            for line in file:
                stripped = line.strip()
                if stripped:
                    first_nonempty_line = stripped
                    break

        if first_nonempty_line.startswith("[") and "image_path:" in first_nonempty_line:
            return cls.parse_detected_inconsistencies_txt(candidate_path)

        return cls.parse_results_txt(candidate_path)

    @staticmethod
    def summarize_results_txt_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate high-level statistics from parsed results.txt entries."""
        if not entries:
            return {
                "num_entries": 0,
                "num_unique_images": 0,
                "num_unique_persons": 0,
                "changed_keypoint_pattern_counts": {},
                "changed_keypoint_index_counts": {},
                "mean_nonzero_shift": 0.0,
                "max_nonzero_shift": 0.0,
                "entries_with_nonzero_shift": 0,
                "top_entries_by_max_shift": [],
            }

        unique_images = {
            entry.get("image_path")
            for entry in entries
            if entry.get("image_path")
        }
        unique_persons = {
            (entry.get("image_path"), entry.get("person_id"))
            for entry in entries
        }

        changed_pattern_counter: Counter = Counter()
        changed_index_counter: Counter = Counter()
        nonzero_shifts: List[float] = []
        ranked_entries: List[Dict[str, Any]] = []

        for entry in entries:
            changed_indices = entry.get("changed_keypoint_indices", []) or []
            pattern_key = tuple(changed_indices)
            changed_pattern_counter[pattern_key] += 1

            for index in changed_indices:
                changed_index_counter[int(index)] += 1

            shift_magnitudes = entry.get("shift_magnitudes", []) or []
            current_nonzero = [float(value) for value in shift_magnitudes if float(value) > 0.0]
            nonzero_shifts.extend(current_nonzero)

            ranked_entries.append(
                {
                    "image_path": entry.get("image_path"),
                    "person_id": entry.get("person_id"),
                    "changed_keypoint_indices": changed_indices,
                    "max_shift": max(current_nonzero) if current_nonzero else 0.0,
                    "mean_shift": float(np.mean(current_nonzero)) if current_nonzero else 0.0,
                    "line_number": entry.get("line_number"),
                }
            )

        ranked_entries.sort(key=lambda item: item["max_shift"], reverse=True)

        return {
            "num_entries": len(entries),
            "num_unique_images": len(unique_images),
            "num_unique_persons": len(unique_persons),
            "changed_keypoint_pattern_counts": {
                str(list(pattern)): count
                for pattern, count in changed_pattern_counter.items()
            },
            "changed_keypoint_index_counts": dict(sorted(changed_index_counter.items())),
            "mean_nonzero_shift": float(np.mean(nonzero_shifts)) if nonzero_shifts else 0.0,
            "max_nonzero_shift": float(max(nonzero_shifts)) if nonzero_shifts else 0.0,
            "entries_with_nonzero_shift": sum(1 for item in ranked_entries if item["max_shift"] > 0.0),
            "top_entries_by_max_shift": ranked_entries[:10],
        }

    @classmethod
    def summarize_results_txt(cls, results_txt_path: str) -> Dict[str, Any]:
        """Parse and summarize a validation results.txt file in one call."""
        return cls.summarize_results_txt_entries(cls.parse_results_txt(results_txt_path))

    @staticmethod
    def _results_txt_entry_identifier(entry: Dict[str, Any]) -> Tuple[Any, Any]:
        """Return the comparison identifier used for results.txt matching."""
        return (
            YPValidation_Test._normalize_results_identifier_path(entry.get("image_path")),
            entry.get("person_id"),
        )

    @classmethod
    def compare_results_txt_entries(
        cls,
        reference_entries: List[Dict[str, Any]],
        candidate_entries: List[Dict[str, Any]],
        match_mode: str = "auto",
    ) -> Dict[str, Any]:
        """Compare two parsed entry lists by normalized image path and optional person id.

        The first list is treated as the reference set. A reference entry counts as
        detected when the second list contains the same identifier. When
        ``match_mode`` is ``auto``, the method compares by ``(image_path, person_id)``
        only if both lists have person identifiers for all entries; otherwise it
        falls back to image-path-only matching. Matching is multiplicity-aware, so
        repeated identifiers are counted up to the number of occurrences present in
        the candidate entries.
        """
        if match_mode not in {"auto", "image_path", "image_path_person_id"}:
            raise ValueError("match_mode must be 'auto', 'image_path', or 'image_path_person_id'")

        if match_mode == "auto":
            candidate_has_person_ids = candidate_entries and all(
                cls._extract_person_identifier(entry) is not None for entry in candidate_entries
            )
            reference_has_person_ids = reference_entries and all(
                cls._extract_person_identifier(entry) is not None for entry in reference_entries
            )
            resolved_match_mode = (
                "image_path_person_id"
                if candidate_has_person_ids and reference_has_person_ids
                else "image_path"
            )
        else:
            resolved_match_mode = match_mode

        reference_identifiers = [
            cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            for entry in reference_entries
        ]
        candidate_identifiers = [
            cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            for entry in candidate_entries
        ]
        remaining_candidate_counts: Counter = Counter(candidate_identifiers)

        detected_entries: List[Dict[str, Any]] = []
        undetected_entries: List[Dict[str, Any]] = []

        for entry in reference_entries:
            image_path = entry.get("image_path")
            person_id = cls._extract_person_identifier(entry)
            identifier = cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)

            entry_summary = {
                "image_path": image_path,
                "person_id": person_id,
                "line_number": entry.get("line_number"),
            }

            if remaining_candidate_counts[identifier] > 0:
                remaining_candidate_counts[identifier] -= 1
                detected_entries.append(entry_summary)
            else:
                undetected_entries.append(dict(entry))

        unmatched_candidate_entries: List[Dict[str, Any]] = []
        consumed_candidate_counts: Counter = Counter(
            cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            for entry in detected_entries
        )

        for entry in candidate_entries:
            identifier = cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            if consumed_candidate_counts[identifier] > 0:
                consumed_candidate_counts[identifier] -= 1
                continue

            if remaining_candidate_counts[identifier] <= 0:
                continue

            unmatched_candidate_entries.append(
                {
                    "image_path": entry.get("image_path"),
                    "person_id": cls._extract_person_identifier(entry),
                    "line_number": entry.get("line_number"),
                }
            )
            remaining_candidate_counts[identifier] -= 1

        detected_count = len(detected_entries)
        total_reference_entries = len(reference_entries)
        effectiveness = (
            float(detected_count / total_reference_entries)
            if total_reference_entries > 0
            else 0.0
        )

        return {
            "total_reference_entries": total_reference_entries,
            "total_candidate_entries": len(candidate_entries),
            "match_mode": resolved_match_mode,
            "detected_count": detected_count,
            "undetected_count": len(undetected_entries),
            "effectiveness": effectiveness,
            "reference_identifiers": reference_identifiers,
            "candidate_identifiers": candidate_identifiers,
            "detected_entries": detected_entries,
            "undetected_entries": undetected_entries,
            "undetected_image_person_pairs": [
                (entry["image_path"], entry["person_id"])
                for entry in undetected_entries
            ],
            "unmatched_candidate_entries": unmatched_candidate_entries,
        }

    @classmethod
    def compare_results_txt(
        cls,
        reference_results_txt_path: str,
        candidate_results_txt_path: str,
    ) -> Dict[str, Any]:
        """Parse and compare two results.txt files by ``image_path`` and ``person_id``."""
        reference_entries = cls.parse_results_txt(reference_results_txt_path)
        candidate_entries = cls.parse_results_txt(candidate_results_txt_path)

        comparison = cls.compare_results_txt_entries(reference_entries, candidate_entries)
        comparison.update(
            {
                "reference_results_txt_path": reference_results_txt_path,
                "candidate_results_txt_path": candidate_results_txt_path,
            }
        )
        return comparison

    @staticmethod
    def _normalize_results_identifier_path(path: Any) -> Any:
        """Normalize a stored path for stable file matching."""
        if path is None:
            return None

        normalized_path = os.path.normcase(os.path.normpath(str(path))).replace('\\', '/')

        for anchor in ('/images/', '/labels/'):
            anchor_position = normalized_path.rfind(anchor)
            if anchor_position != -1:
                return normalized_path[anchor_position + 1:]

        return normalized_path

    @staticmethod
    def _extract_person_identifier(entry: Mapping[str, Any]) -> Any:
        """Return a normalized person identifier from supported field names."""
        person_identifier = entry.get("person_id")
        if person_identifier is None:
            person_identifier = entry.get("person_index")
        if person_identifier is None:
            person_identifier = entry.get("person_idx")

        if person_identifier is None:
            return None

        try:
            return int(person_identifier)
        except (TypeError, ValueError):
            return person_identifier

    @classmethod
    def _distance_anomaly_entry_identifier(
        cls,
        entry: Dict[str, Any],
        match_mode: str,
    ) -> Any:
        """Return the identifier used when matching ranked anomaly entries."""
        image_path = cls._normalize_results_identifier_path(entry.get("image_path"))
        if match_mode == "image_path_person_id":
            return image_path, cls._extract_person_identifier(entry)
        return image_path

    @staticmethod
    def _extract_distance_anomaly_entries(raw_object: Any) -> Optional[List[Dict[str, Any]]]:
        """Return a list of anomaly entries when a supported object shape is provided."""
        if isinstance(raw_object, dict):
            if isinstance(raw_object.get("images"), list):
                return [dict(entry) for entry in raw_object["images"] if isinstance(entry, dict)]
            if isinstance(raw_object.get("records"), list):
                return [dict(entry) for entry in raw_object["records"] if isinstance(entry, dict)]
            if "image_path" in raw_object:
                return [dict(raw_object)]

        if isinstance(raw_object, list):
            if all(isinstance(entry, dict) for entry in raw_object):
                return [dict(entry) for entry in raw_object]

        return None

    @classmethod
    def _normalize_distance_anomaly_entries(
        cls,
        entries: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Normalize ranked anomaly entries to a consistent serializable schema."""
        normalized_entries: List[Dict[str, Any]] = []

        for fallback_rank, entry in enumerate(entries, start=1):
            normalized_entry = dict(entry)
            normalized_entry["image_path"] = entry.get("image_path")
            normalized_entry["person_id"] = cls._extract_person_identifier(entry)

            if normalized_entry.get("rank") is None:
                normalized_entry["rank"] = fallback_rank
            else:
                try:
                    normalized_entry["rank"] = int(normalized_entry["rank"])
                except (TypeError, ValueError):
                    normalized_entry["rank"] = fallback_rank

            if entry.get("person_index") is not None:
                try:
                    normalized_entry["person_index"] = int(entry.get("person_index"))
                except (TypeError, ValueError):
                    normalized_entry["person_index"] = entry.get("person_index")

            if entry.get("person_idx") is not None:
                try:
                    normalized_entry["person_idx"] = int(entry.get("person_idx"))
                except (TypeError, ValueError):
                    normalized_entry["person_idx"] = entry.get("person_idx")

            if normalized_entry.get("error") is not None:
                try:
                    normalized_entry["error"] = float(normalized_entry["error"])
                except (TypeError, ValueError):
                    pass

            for score_key in ("max_anomaly_score", "mean_anomaly_score", "sort_score"):
                if normalized_entry.get(score_key) is None:
                    continue
                try:
                    normalized_entry[score_key] = float(normalized_entry[score_key])
                except (TypeError, ValueError):
                    pass

            if normalized_entry.get("is_anomaly") is not None:
                normalized_entry["is_anomaly"] = bool(normalized_entry["is_anomaly"])

            normalized_entries.append(normalized_entry)

        normalized_entries.sort(key=lambda entry: entry.get("rank", float("inf")))
        for fallback_rank, entry in enumerate(normalized_entries, start=1):
            entry["rank"] = fallback_rank if entry.get("rank") is None else int(entry["rank"])

        return normalized_entries

    @classmethod
    def parse_distance_anomaly_report(cls, distance_anomaly_report_path: str) -> List[Dict[str, Any]]:
        """Parse a saved distance-anomaly ranking file into ranked entries.

        Supported formats:
        - a Python literal dict containing an ``images`` list
        - a Python literal list of dicts
        - one dict per line
        - one key-value line per entry in the style ``key: value | key: value``
        """
        with open(distance_anomaly_report_path, "r", encoding="utf-8") as file_handle:
            raw_text = file_handle.read().strip()

        if not raw_text:
            return []

        try:
            parsed_object = ast.literal_eval(raw_text)
        except (ValueError, SyntaxError):
            parsed_object = None

        extracted_entries = cls._extract_distance_anomaly_entries(parsed_object)
        if extracted_entries is not None:
            return cls._normalize_distance_anomaly_entries(extracted_entries)

        parsed_entries: List[Dict[str, Any]] = []
        for line_number, line in enumerate(raw_text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                if stripped.startswith("{") and stripped.endswith("}"):
                    parsed_line = ast.literal_eval(stripped)
                    if not isinstance(parsed_line, dict):
                        raise ValueError("Parsed line is not a dictionary")
                else:
                    parsed_line = cls.parse_results_txt_line(stripped)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(
                    f"Failed to parse distance anomaly report {distance_anomaly_report_path} at line {line_number}: {exc}"
                ) from exc

            parsed_line["line_number"] = line_number
            parsed_entries.append(parsed_line)

        return cls._normalize_distance_anomaly_entries(parsed_entries)

    @classmethod
    def save_distance_anomaly_report(
        cls,
        distance_anomaly_results: Dict[str, Any],
        output_path: str,
        only_anomalies: bool = False,
    ) -> str:
        """Save ranked distance-anomaly results to a line-based text file.

        The saved format is designed to be parseable by
        ``parse_distance_anomaly_report()`` and stable across notebooks.
        """
        entries = cls._extract_distance_anomaly_entries(distance_anomaly_results)
        if entries is None:
            raise TypeError(
                "distance_anomaly_results must contain an 'images' list, a 'records' list, "
                "a single record dictionary, or be a list of dictionaries"
            )

        normalized_entries = cls._normalize_distance_anomaly_entries(entries)

        output_path_abs = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path_abs), exist_ok=True)

        serialized_lines: List[str] = []
        for fallback_rank, image_entry in enumerate(normalized_entries, start=1):
            if only_anomalies and not bool(image_entry.get("is_anomaly")):
                continue

            person_identifier = cls._extract_person_identifier(image_entry)
            error_value = image_entry.get("error")
            if error_value is None:
                error_value = image_entry.get("sort_score")
            if error_value is None:
                error_value = image_entry.get("max_anomaly_score")
            if error_value is None:
                error_value = image_entry.get("mean_anomaly_score")

            serialized_error = None
            if error_value is not None:
                serialized_error = float(error_value)
                if not np.isfinite(serialized_error):
                    serialized_error = None

            serialized_entry = {
                "rank": int(image_entry.get("rank", fallback_rank)),
                "image_path": image_entry.get("image_path"),
                "person_id": person_identifier,
                "error": serialized_error,
                "is_anomaly": bool(image_entry.get("is_anomaly")),
            }
            if image_entry.get("person_index") is not None:
                serialized_entry["person_index"] = image_entry.get("person_index")
            if image_entry.get("person_idx") is not None:
                serialized_entry["person_idx"] = image_entry.get("person_idx")
            if image_entry.get("max_anomaly_score") is not None:
                max_anomaly_score = float(image_entry["max_anomaly_score"])
                serialized_entry["max_anomaly_score"] = (
                    max_anomaly_score if np.isfinite(max_anomaly_score) else None
                )
            if image_entry.get("mean_anomaly_score") is not None:
                mean_anomaly_score = float(image_entry["mean_anomaly_score"])
                serialized_entry["mean_anomaly_score"] = (
                    mean_anomaly_score if np.isfinite(mean_anomaly_score) else None
                )
            serialized_lines.append(repr(serialized_entry))

        with open(output_path_abs, "w", encoding="utf-8") as file_handle:
            file_handle.write("\n".join(serialized_lines) + ("\n" if serialized_lines else ""))

        return output_path_abs

    @classmethod
    def compare_results_txt_to_distance_anomalies(
        cls,
        reference_results_txt_path: str,
        distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        top_k: int,
        match_mode: str = "auto",
        include_reference_fields: bool = False,
        reference_field_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compare a reference ``results.txt`` file with a ranked anomaly report.

        Args:
            reference_results_txt_path: Path to the reference ``results.txt`` file.
            distance_anomaly_report_path: Path to a saved anomaly ranking or an
                in-memory ranking object compatible with ``parse_distance_anomaly_report``.
            top_k: Number of top-ranked anomaly entries to inspect.
            match_mode: ``auto``, ``image_path``, or ``image_path_person_id``.
            include_reference_fields: When ``True``, add per-reference entries
                with selected fields from ``results.txt`` and flags describing
                whether the entry was found in the report and inside ``top_k``.
            reference_field_names: Optional subset of ``results.txt`` fields to
                include when ``include_reference_fields`` is enabled. When omitted,
                all reference fields except identifiers are included.

        Returns:
            Serializable dictionary containing the overlap summary and the exact
            matched entries. The ``summary_text`` field contains a ready-to-print
            sentence in Polish.
        """
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if match_mode not in {"auto", "image_path", "image_path_person_id"}:
            raise ValueError("match_mode must be 'auto', 'image_path', or 'image_path_person_id'")

        reference_entries = cls.parse_results_txt(reference_results_txt_path)
        ranked_entries: List[Dict[str, Any]]
        if isinstance(distance_anomaly_report_path, (str, os.PathLike)):
            ranked_entries = cls.parse_distance_anomaly_report(os.fspath(distance_anomaly_report_path))
            ranked_report_reference = os.path.abspath(os.fspath(distance_anomaly_report_path))
        else:
            extracted_entries = cls._extract_distance_anomaly_entries(distance_anomaly_report_path)
            if extracted_entries is None:
                raise TypeError(
                    "distance_anomaly_report_path must be a path or an anomaly result object "
                    "containing 'images'/'records' entries"
                )
            ranked_entries = cls._normalize_distance_anomaly_entries(extracted_entries)
            ranked_report_reference = "<in-memory>"

        if match_mode == "auto":
            candidate_has_person_ids = ranked_entries and all(
                cls._extract_person_identifier(entry) is not None for entry in ranked_entries
            )
            reference_has_person_ids = reference_entries and all(
                cls._extract_person_identifier(entry) is not None for entry in reference_entries
            )
            resolved_match_mode = (
                "image_path_person_id"
                if candidate_has_person_ids and reference_has_person_ids
                else "image_path"
            )
        else:
            resolved_match_mode = match_mode

        reference_identifiers = [
            cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            for entry in reference_entries
        ]
        reference_identifier_counts = Counter(reference_identifiers)

        ranked_entries_sorted = sorted(ranked_entries, key=lambda entry: entry.get("rank", float("inf")))
        top_entries = ranked_entries_sorted[:top_k]

        top_entries_by_identifier: Dict[Any, List[Dict[str, Any]]] = {}
        report_entries_by_identifier: Dict[Any, List[Dict[str, Any]]] = {}

        for entry in ranked_entries_sorted:
            identifier = cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            report_entries_by_identifier.setdefault(identifier, []).append(entry)

        for entry in top_entries:
            identifier = cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            top_entries_by_identifier.setdefault(identifier, []).append(entry)

        requested_reference_fields = list(reference_field_names or [])
        reference_report_entries: List[Dict[str, Any]] = []
        matched_top_entries: List[Dict[str, Any]] = []
        matched_overall_entries: List[Dict[str, Any]] = []

        for reference_entry in reference_entries:
            identifier = cls._distance_anomaly_entry_identifier(reference_entry, resolved_match_mode)
            report_match_bucket = report_entries_by_identifier.get(identifier, [])
            top_match_bucket = top_entries_by_identifier.get(identifier, [])

            report_match = report_match_bucket.pop(0) if report_match_bucket else None
            top_match = top_match_bucket.pop(0) if top_match_bucket else None

            extracted_entry = {
                "image_path": reference_entry.get("image_path"),
                "person_id": cls._extract_person_identifier(reference_entry),
                "found_in_report": report_match is not None,
                "found_in_top_k": top_match is not None,
                "report_rank": report_match.get("rank") if report_match is not None else None,
                "report_error": report_match.get("error") if report_match is not None else None,
                "report_is_anomaly": report_match.get("is_anomaly") if report_match is not None else None,
                "reference_line_number": reference_entry.get("line_number"),
            }

            if requested_reference_fields:
                for field_name in requested_reference_fields:
                    extracted_entry[field_name] = reference_entry.get(field_name)
            else:
                for field_name, field_value in reference_entry.items():
                    if field_name in {"image_path", "person_id", "line_number"}:
                        continue
                    extracted_entry[field_name] = field_value

            reference_report_entries.append(extracted_entry)

            if top_match is not None:
                matched_top_entries.append(
                    {
                        "rank": top_match.get("rank"),
                        "image_path": reference_entry.get("image_path"),
                        "person_id": cls._extract_person_identifier(reference_entry),
                        "error": top_match.get("error"),
                        "is_anomaly": top_match.get("is_anomaly"),
                    }
                )

            if report_match is not None:
                matched_overall_entries.append(
                    {
                        "rank": report_match.get("rank"),
                        "image_path": reference_entry.get("image_path"),
                        "person_id": cls._extract_person_identifier(reference_entry),
                        "error": report_match.get("error"),
                        "is_anomaly": report_match.get("is_anomaly"),
                    }
                )

        matched_top_count = len(matched_top_entries)
        last_matched_rank_in_top_k = max(
            (int(entry["rank"]) for entry in matched_top_entries if entry.get("rank") is not None),
            default=None,
        )
        best_rank_overall = min(
            (int(entry["rank"]) for entry in matched_overall_entries if entry.get("rank") is not None),
            default=None,
        )
        worst_rank_overall = max(
            (int(entry["rank"]) for entry in matched_overall_entries if entry.get("rank") is not None),
            default=None,
        )
        matched_overall_count = len(matched_overall_entries)
        avg_rank_all_reference = (
            sum(int(entry["rank"]) for entry in matched_overall_entries if entry.get("rank") is not None)
            / matched_overall_count
            if matched_overall_count > 0
            else None
        )

        if last_matched_rank_in_top_k is None:
            summary_text = (
                f"W top {top_k} pozycji rankingu anomalii znajduje się 0 poprawnych plików "
                f"(plików z results.txt); żaden z nich nie występuje w badanym zakresie."
            )
        else:
            avg_str = f"{avg_rank_all_reference:.1f}" if avg_rank_all_reference is not None else "brak"
            summary_text = (
                f"W top {top_k} pozycji rankingu anomalii znajduje się {matched_top_count} poprawnych plików "
                f"(plików z results.txt), a ostatni z nich zajmuje pozycję {last_matched_rank_in_top_k}. "
                f"Średnia pozycja wszystkich poszukiwanych: {avg_str}."
            )

        comparison = {
            "reference_results_txt_path": os.path.abspath(reference_results_txt_path),
            "distance_anomaly_report_path": ranked_report_reference,
            "top_k": int(top_k),
            "match_mode": resolved_match_mode,
            "total_reference_entries": len(reference_entries),
            "total_ranked_entries": len(ranked_entries_sorted),
            "matched_top_k_count": matched_top_count,
            "last_matched_rank_in_top_k": last_matched_rank_in_top_k,
            "matched_top_k_entries": matched_top_entries,
            "matched_overall_count": matched_overall_count,
            "avg_rank_all_reference": avg_rank_all_reference,
            "best_rank_overall": best_rank_overall,
            "worst_rank_overall": worst_rank_overall,
            "matched_overall_entries": matched_overall_entries,
            "summary_text": summary_text,
        }

        if include_reference_fields:
            comparison["reference_report_entries"] = reference_report_entries
            comparison["reference_report_found_count"] = sum(
                1 for entry in reference_report_entries if entry["found_in_report"]
            )
            comparison["reference_top_k_found_count"] = sum(
                1 for entry in reference_report_entries if entry["found_in_top_k"]
            )

        return comparison

    @classmethod
    def compare_results_txt_to_distance_anomalies_with_outside_keypoints(
        cls,
        reference_results_txt_path: str,
        distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
        outside_keypoint_records: Union[List[Dict[str, Any]], Dict[str, Any]],
        top_k: int,
        match_mode: str = "auto",
    ) -> Dict[str, Any]:
        """Compare top-k anomaly matches and then add matches from outside-mask keypoints.

        This variant preserves the default ranking-based comparison and only adds
        supplemental matches for persons present in ``outside_keypoint_records``
        that were not already matched within ``top_k``.
        """
        base_comparison = cls.compare_results_txt_to_distance_anomalies(
            reference_results_txt_path=reference_results_txt_path,
            distance_anomaly_report_path=distance_anomaly_report_path,
            top_k=top_k,
            match_mode=match_mode,
        )

        if isinstance(outside_keypoint_records, dict):
            if isinstance(outside_keypoint_records.get("outside_keypoints"), list):
                raw_outside_entries = outside_keypoint_records["outside_keypoints"]
            elif isinstance(outside_keypoint_records.get("records"), list):
                raw_outside_entries = outside_keypoint_records["records"]
            else:
                raise TypeError(
                    "outside_keypoint_records must be a list of dictionaries or contain "
                    "an 'outside_keypoints'/'records' list"
                )
        elif isinstance(outside_keypoint_records, list):
            raw_outside_entries = outside_keypoint_records
        else:
            raise TypeError(
                "outside_keypoint_records must be a list of dictionaries or a dictionary"
            )

        reference_entries = cls.parse_results_txt(reference_results_txt_path)
        resolved_match_mode = str(base_comparison["match_mode"])
        reference_identifiers = [
            cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            for entry in reference_entries
        ]
        remaining_counts = Counter(reference_identifiers)

        for entry in base_comparison.get("matched_top_k_entries", []):
            identifier = cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            if remaining_counts[identifier] > 0:
                remaining_counts[identifier] -= 1

        grouped_outside_entries: Dict[Any, Dict[str, Any]] = {}
        for entry in raw_outside_entries:
            if not isinstance(entry, dict):
                continue

            identifier = cls._distance_anomaly_entry_identifier(entry, resolved_match_mode)
            if identifier is None:
                continue

            grouped_entry = grouped_outside_entries.get(identifier)
            if grouped_entry is None:
                grouped_entry = {
                    "image_path": entry.get("image_path"),
                    "person_id": cls._extract_person_identifier(entry),
                    "outside_keypoint_indices": [],
                    "outside_keypoint_count": 0,
                }
                grouped_outside_entries[identifier] = grouped_entry

            keypoint_index = entry.get("keypoint_index")
            if keypoint_index is not None:
                try:
                    normalized_index = int(keypoint_index)
                except (TypeError, ValueError):
                    normalized_index = keypoint_index
                if normalized_index not in grouped_entry["outside_keypoint_indices"]:
                    grouped_entry["outside_keypoint_indices"].append(normalized_index)
            grouped_entry["outside_keypoint_count"] = len(grouped_entry["outside_keypoint_indices"])

        supplemental_outside_matches: List[Dict[str, Any]] = []
        for identifier, entry in grouped_outside_entries.items():
            if remaining_counts[identifier] <= 0:
                continue
            remaining_counts[identifier] -= 1
            supplemental_outside_matches.append(entry)

        supplemental_outside_matches.sort(
            key=lambda entry: (
                -int(entry.get("outside_keypoint_count", 0)),
                str(entry.get("image_path")),
                str(entry.get("person_id")),
            )
        )

        matched_top_k_plus_outside_count = (
            int(base_comparison.get("matched_top_k_count", 0))
            + len(supplemental_outside_matches)
        )

        summary_text = str(base_comparison.get("summary_text", ""))
        if supplemental_outside_matches:
            summary_text += (
                f" Po doliczeniu {len(supplemental_outside_matches)} rekordów z visible keypoints "
                f"outside mask liczba trafień rośnie do {matched_top_k_plus_outside_count}."
            )

        summary_text += (
            f" Łącznie wykryto {matched_top_k_plus_outside_count} plików z "
            f"{len(reference_entries)} plików referencyjnych."
        )

        comparison = dict(base_comparison)
        comparison["matched_top_k_plus_outside_keypoints_count"] = matched_top_k_plus_outside_count
        comparison["detected_files_count"] = matched_top_k_plus_outside_count
        comparison["reference_files_count"] = len(reference_entries)
        comparison["supplemental_outside_keypoint_matches"] = supplemental_outside_matches
        comparison["summary_text"] = summary_text
        return comparison

    @staticmethod
    def _normalize_lookup_path(path: str) -> str:
        """Normalize a path for stable cross-dataset matching."""
        return os.path.normcase(os.path.abspath(path)).replace('\\', '/')

    @staticmethod
    def _normalize_relative_lookup_path(path: str, dataset_root: str) -> Optional[str]:
        """Resolve a normalized dataset-relative path when possible."""
        try:
            relative_path = os.path.relpath(path, dataset_root) if os.path.isabs(path) else path
        except ValueError:
            return None

        normalized_relative_path = os.path.normpath(relative_path).replace('\\', '/')
        if normalized_relative_path == '..' or normalized_relative_path.startswith('../'):
            return None
        return normalized_relative_path

    @staticmethod
    def _format_yolo_float(value: float) -> str:
        """Format YOLO numeric values deterministically."""
        return f"{float(value):.6f}"

    @staticmethod
    def _load_yolo_pose_label_entries(label_path: str) -> List[Dict[str, Any]]:
        """Load a YOLO pose label file into structured entries."""
        from .image_validation import _parse_yolo_keypoint_values

        entries: List[Dict[str, Any]] = []
        with open(label_path, 'r', encoding='utf-8') as file_handle:
            for line in file_handle:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                parts = stripped_line.split()
                if len(parts) < 5:
                    continue

                class_id = parts[0]
                bbox = [float(value) for value in parts[1:5]]
                keypoint_values = [float(value) for value in parts[5:]]
                keypoints = _parse_yolo_keypoint_values(
                    keypoint_values,
                    include_visibility=True,
                    keypoint_format='auto',
                )
                entries.append(
                    {
                        'class_id': class_id,
                        'bbox': bbox,
                        'keypoints': keypoints,
                    }
                )

        return entries

    @classmethod
    def _write_yolo_pose_label_entries(cls, label_path: str, entries: List[Dict[str, Any]]) -> None:
        """Write structured YOLO pose entries back to a label file."""
        serialized_lines: List[str] = []

        for entry in entries:
            bbox = [cls._format_yolo_float(value) for value in entry['bbox']]
            keypoints = np.asarray(entry['keypoints'], dtype=float)
            keypoint_values = [cls._format_yolo_float(value) for value in keypoints.reshape(-1)]
            serialized_lines.append(" ".join([str(entry['class_id'])] + bbox + keypoint_values))

        with open(label_path, 'w', encoding='utf-8') as file_handle:
            file_handle.write("\n".join(serialized_lines) + ("\n" if serialized_lines else ""))

    @staticmethod
    def _clone_yolo_pose_label_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Return a deep copy of a structured YOLO pose label entry."""
        return {
            'class_id': str(entry['class_id']),
            'bbox': [float(value) for value in entry['bbox']],
            'keypoints': np.asarray(entry['keypoints'], dtype=float).copy(),
        }

    @staticmethod
    def _basename_lookup_key(path: str) -> str:
        """Return a normalized basename key for cross-dataset fallback matching."""
        return os.path.normcase(os.path.basename(path))

    @classmethod
    def _build_dataset_person_lookups(
        cls,
        dataset: 'YOLOPoseDataset',
    ) -> Tuple[
        Dict[Tuple[str, int], Dict[str, Any]],
        Dict[Tuple[str, int], Dict[str, Any]],
        Dict[Tuple[str, int], Optional[Dict[str, Any]]],
    ]:
        """Build person lookup tables for a YOLOPoseDataset."""
        by_absolute_path: Dict[Tuple[str, int], Dict[str, Any]] = {}
        by_relative_path: Dict[Tuple[str, int], Dict[str, Any]] = {}
        by_basename: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}

        for sample in dataset:
            image_path = str(sample['image_path'])
            label_path = str(sample['label_path'])
            image_rel_path = os.path.relpath(image_path, dataset.dataset_dir).replace('\\', '/')
            label_name = os.path.basename(label_path)

            for person_id, keypoints in enumerate(sample['all_keypoints']):
                person_record = {
                    'image_path': image_path,
                    'image_rel_path': image_rel_path,
                    'label_path': label_path,
                    'label_name': label_name,
                    'person_id': int(person_id),
                    'keypoints': np.asarray(keypoints, dtype=float),
                }
                by_absolute_path[(cls._normalize_lookup_path(image_path), int(person_id))] = person_record
                by_relative_path[(image_rel_path, int(person_id))] = person_record
                basename_key = (cls._basename_lookup_key(image_path), int(person_id))
                if basename_key in by_basename:
                    by_basename[basename_key] = None
                else:
                    by_basename[basename_key] = person_record

        return by_absolute_path, by_relative_path, by_basename

    @classmethod
    def _build_dataset_image_lookups(
        cls,
        dataset: 'YOLOPoseDataset',
    ) -> Tuple[
        Dict[str, Dict[str, Any]],
        Dict[str, Dict[str, Any]],
        Dict[str, Optional[Dict[str, Any]]],
    ]:
        """Build image lookup tables for a YOLOPoseDataset."""
        by_absolute_path: Dict[str, Dict[str, Any]] = {}
        by_relative_path: Dict[str, Dict[str, Any]] = {}
        by_basename: Dict[str, Optional[Dict[str, Any]]] = {}

        for sample in dataset:
            image_path = str(sample['image_path'])
            label_path = str(sample['label_path'])
            image_rel_path = os.path.relpath(image_path, dataset.dataset_dir).replace('\\', '/')
            image_record = {
                'image_path': image_path,
                'image_rel_path': image_rel_path,
                'label_path': label_path,
                'label_name': os.path.basename(label_path),
                'person_count': len(sample['all_keypoints']),
            }
            by_absolute_path[cls._normalize_lookup_path(image_path)] = image_record
            by_relative_path[image_rel_path] = image_record
            image_basename = cls._basename_lookup_key(image_path)
            if image_basename in by_basename:
                by_basename[image_basename] = None
            else:
                by_basename[image_basename] = image_record

        return by_absolute_path, by_relative_path, by_basename

    @classmethod
    def _resolve_person_record(
        cls,
        entry: Dict[str, Any],
        dataset: 'YOLOPoseDataset',
        by_absolute_path: Dict[Tuple[str, int], Dict[str, Any]],
        by_relative_path: Dict[Tuple[str, int], Dict[str, Any]],
        by_basename: Dict[Tuple[str, int], Optional[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Resolve a parsed results entry to a person record in the dataset."""
        image_path = entry.get('image_path')
        person_id = entry.get('person_id')
        if image_path is None or person_id is None:
            return None

        absolute_key = (cls._normalize_lookup_path(str(image_path)), int(person_id))
        if absolute_key in by_absolute_path:
            return by_absolute_path[absolute_key]

        relative_path = cls._normalize_relative_lookup_path(str(image_path), dataset.dataset_dir)
        if relative_path is not None:
            relative_match = by_relative_path.get((relative_path, int(person_id)))
            if relative_match is not None:
                return relative_match

        basename_match = by_basename.get((cls._basename_lookup_key(str(image_path)), int(person_id)))
        if basename_match is not None:
            return basename_match

        return None

    @classmethod
    def _resolve_target_image_record(
        cls,
        entry: Dict[str, Any],
        dataset: 'YOLOPoseDataset',
        by_absolute_path: Dict[str, Dict[str, Any]],
        by_relative_path: Dict[str, Dict[str, Any]],
        by_basename: Dict[str, Optional[Dict[str, Any]]],
        fallback_relative_path: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Resolve a parsed results entry to an image record in the target dataset."""
        image_path = entry.get('image_path')
        if image_path is not None:
            absolute_key = cls._normalize_lookup_path(str(image_path))
            if absolute_key in by_absolute_path:
                return by_absolute_path[absolute_key]

            relative_path = cls._normalize_relative_lookup_path(str(image_path), dataset.dataset_dir)
            if relative_path is not None and relative_path in by_relative_path:
                return by_relative_path[relative_path]

            basename_match = by_basename.get(cls._basename_lookup_key(str(image_path)))
            if basename_match is not None:
                return basename_match

        if fallback_relative_path is not None:
            relative_match = by_relative_path.get(fallback_relative_path.replace('\\', '/'))
            if relative_match is not None:
                return relative_match

            basename_match = by_basename.get(cls._basename_lookup_key(str(fallback_relative_path)))
            if basename_match is not None:
                return basename_match

        return None

    @classmethod
    def copy_missing_person_labels_from_results(
        cls,
        reference_results_txt_path: str,
        candidate_results_txt_path: str,
        reference_dataset: 'YOLOPoseDataset',
        target_dataset: 'YOLOPoseDataset',
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Copy missing person labels from dataset 1 into dataset 2 based on results.txt comparison.

        Workflow:
        1. Read both results.txt files and compare ``(image_path, person_id)`` pairs.
              When ``top_k`` is provided, only the first ``top_k`` candidate entries
              from ``candidate_results_txt_path`` are considered detected.
        2. For records present in the first file but missing in the second, find the
           matching person label in ``reference_dataset``.
        3. Find the corresponding image in ``target_dataset``.
        4. If the image exists and the target label file contains that ``person_id``,
           replace that one label line with the line from ``reference_dataset``.
        5. If the image is missing in ``target_dataset``, or that ``person_id`` does
           not exist there, skip the copy.

        Both ``reference_dataset`` and ``target_dataset`` must be instances of
        ``YOLOPoseDataset``.
        """
        from ...datasets.yolo_pose_dataset import YOLOPoseDataset

        if not isinstance(reference_dataset, YOLOPoseDataset):
            raise TypeError("reference_dataset must be a YOLOPoseDataset instance")
        if not isinstance(target_dataset, YOLOPoseDataset):
            raise TypeError("target_dataset must be a YOLOPoseDataset instance")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0 when provided")

        if top_k is None:
            comparison = cls.compare_results_txt(
                reference_results_txt_path,
                candidate_results_txt_path,
            )
        else:
            reference_entries = cls.parse_results_txt(reference_results_txt_path)
            candidate_entries = cls.parse_candidate_results_file(candidate_results_txt_path)[:top_k]
            comparison = cls.compare_results_txt_entries(reference_entries, candidate_entries)
            comparison.update(
                {
                    'reference_results_txt_path': reference_results_txt_path,
                    'candidate_results_txt_path': candidate_results_txt_path,
                    'candidate_top_k': top_k,
                }
            )

        missing_reference_entries = comparison['undetected_entries']

        source_person_lookup_abs, source_person_lookup_rel, source_person_lookup_basename = cls._build_dataset_person_lookups(reference_dataset)
        target_image_lookup_abs, target_image_lookup_rel, target_image_lookup_basename = cls._build_dataset_image_lookups(target_dataset)

        source_label_entries_cache: Dict[str, List[Dict[str, Any]]] = {}
        target_label_entries_cache: Dict[str, List[Dict[str, Any]]] = {}

        copied_entries: List[Dict[str, Any]] = []
        skipped_missing_source_person: List[Dict[str, Any]] = []
        skipped_missing_target_image: List[Dict[str, Any]] = []
        skipped_missing_target_person: List[Dict[str, Any]] = []

        for missing_entry in missing_reference_entries:
            source_person = cls._resolve_person_record(
                missing_entry,
                dataset=reference_dataset,
                by_absolute_path=source_person_lookup_abs,
                by_relative_path=source_person_lookup_rel,
                by_basename=source_person_lookup_basename,
            )
            if source_person is None:
                skipped_missing_source_person.append(dict(missing_entry))
                continue

            target_image = cls._resolve_target_image_record(
                missing_entry,
                dataset=target_dataset,
                by_absolute_path=target_image_lookup_abs,
                by_relative_path=target_image_lookup_rel,
                by_basename=target_image_lookup_basename,
                fallback_relative_path=str(source_person['image_rel_path']),
            )
            if target_image is None:
                skipped_missing_target_image.append(dict(missing_entry))
                continue

            person_id = int(missing_entry['person_id'])
            target_label_path = str(target_image['label_path'])
            if target_label_path not in target_label_entries_cache:
                target_label_entries_cache[target_label_path] = cls._load_yolo_pose_label_entries(target_label_path)

            target_entries = target_label_entries_cache[target_label_path]
            if person_id < 0 or person_id >= len(target_entries):
                skipped_missing_target_person.append(
                    {
                        **dict(missing_entry),
                        'target_image_path': target_image['image_path'],
                        'target_label_path': target_label_path,
                        'target_person_count': len(target_entries),
                    }
                )
                continue

            source_label_path = str(source_person['label_path'])
            if source_label_path not in source_label_entries_cache:
                source_label_entries_cache[source_label_path] = cls._load_yolo_pose_label_entries(source_label_path)

            source_entries = source_label_entries_cache[source_label_path]
            if person_id < 0 or person_id >= len(source_entries):
                skipped_missing_source_person.append(dict(missing_entry))
                continue

            target_entries[person_id] = cls._clone_yolo_pose_label_entry(source_entries[person_id])
            copied_entries.append(
                {
                    'image_path': target_image['image_path'],
                    'label_path': target_label_path,
                    'person_id': person_id,
                    'source_image_path': source_person['image_path'],
                    'source_label_path': source_label_path,
                    'reference_entry': dict(missing_entry),
                }
            )

        for target_label_path, target_entries in target_label_entries_cache.items():
            cls._write_yolo_pose_label_entries(target_label_path, target_entries)

        if hasattr(target_dataset, '_validate_pairs') and hasattr(target_dataset, '_valid_pairs'):
            target_dataset._valid_pairs = target_dataset._validate_pairs()

        missing_reference_count = len(missing_reference_entries)
        copy_effectiveness = (
            float(len(copied_entries) / missing_reference_count)
            if missing_reference_count > 0
            else 0.0
        )

        return {
            'reference_results_txt_path': os.path.abspath(reference_results_txt_path),
            'candidate_results_txt_path': os.path.abspath(candidate_results_txt_path),
            'candidate_top_k': top_k,
            'comparison': comparison,
            'missing_reference_entries': missing_reference_entries,
            'missing_reference_count': missing_reference_count,
            'copied_entries': copied_entries,
            'copied_count': len(copied_entries),
            'copy_effectiveness': copy_effectiveness,
            'skipped_missing_source_person': skipped_missing_source_person,
            'skipped_missing_target_image': skipped_missing_target_image,
            'skipped_missing_target_person': skipped_missing_target_person,
        }

    def summary(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the current test state."""
        return {
            "name": self.name,
            "metadata": dict(self.metadata),
            "num_results": len(self._results),
            "results": self.results,
        }


def parse_results_txt_line(line: str) -> Dict[str, Any]:
    """Module-level wrapper for YPValidation_Test.parse_results_txt_line."""
    return YPValidation_Test.parse_results_txt_line(line)


def parse_results_txt(results_txt_path: str) -> List[Dict[str, Any]]:
    """Module-level wrapper for YPValidation_Test.parse_results_txt."""
    return YPValidation_Test.parse_results_txt(results_txt_path)


def summarize_results_txt_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Module-level wrapper for YPValidation_Test.summarize_results_txt_entries."""
    return YPValidation_Test.summarize_results_txt_entries(entries)


def summarize_results_txt(results_txt_path: str) -> Dict[str, Any]:
    """Module-level wrapper for YPValidation_Test.summarize_results_txt."""
    return YPValidation_Test.summarize_results_txt(results_txt_path)


def parse_distance_anomaly_report(distance_anomaly_report_path: str) -> List[Dict[str, Any]]:
    """Module-level wrapper for YPValidation_Test.parse_distance_anomaly_report."""
    return YPValidation_Test.parse_distance_anomaly_report(distance_anomaly_report_path)


def save_distance_anomaly_report(
    distance_anomaly_results: Dict[str, Any],
    output_path: str,
    only_anomalies: bool = False,
) -> str:
    """Module-level wrapper for YPValidation_Test.save_distance_anomaly_report."""
    return YPValidation_Test.save_distance_anomaly_report(
        distance_anomaly_results,
        output_path,
        only_anomalies=only_anomalies,
    )


def compare_results_txt_to_distance_anomalies(
    reference_results_txt_path: str,
    distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    top_k: int,
    match_mode: str = "auto",
    include_reference_fields: bool = False,
    reference_field_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Module-level wrapper for YPValidation_Test.compare_results_txt_to_distance_anomalies."""
    return YPValidation_Test.compare_results_txt_to_distance_anomalies(
        reference_results_txt_path,
        distance_anomaly_report_path,
        top_k,
        match_mode=match_mode,
        include_reference_fields=include_reference_fields,
        reference_field_names=reference_field_names,
    )


def compare_results_txt_to_distance_anomalies_with_outside_keypoints(
    reference_results_txt_path: str,
    distance_anomaly_report_path: Union[str, Dict[str, Any], List[Dict[str, Any]]],
    outside_keypoint_records: Union[List[Dict[str, Any]], Dict[str, Any]],
    top_k: int,
    match_mode: str = "auto",
) -> Dict[str, Any]:
    """Module-level wrapper for the outside-keypoint-aware anomaly comparison."""
    return YPValidation_Test.compare_results_txt_to_distance_anomalies_with_outside_keypoints(
        reference_results_txt_path,
        distance_anomaly_report_path,
        outside_keypoint_records,
        top_k,
        match_mode=match_mode,
    )


def compare_results_txt_entries(
    reference_entries: List[Dict[str, Any]],
    candidate_entries: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Module-level wrapper for YPValidation_Test.compare_results_txt_entries."""
    return YPValidation_Test.compare_results_txt_entries(reference_entries, candidate_entries)


def compare_results_txt(
    reference_results_txt_path: str,
    candidate_results_txt_path: str,
) -> Dict[str, Any]:
    """Module-level wrapper for YPValidation_Test.compare_results_txt."""
    return YPValidation_Test.compare_results_txt(
        reference_results_txt_path,
        candidate_results_txt_path,
    )


def copy_missing_person_labels_from_results(
    reference_results_txt_path: str,
    candidate_results_txt_path: str,
    reference_dataset: 'YOLOPoseDataset',
    target_dataset: 'YOLOPoseDataset',
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Module-level wrapper for YPValidation_Test.copy_missing_person_labels_from_results."""
    return YPValidation_Test.copy_missing_person_labels_from_results(
        reference_results_txt_path,
        candidate_results_txt_path,
        reference_dataset,
        target_dataset,
        top_k=top_k,
    )
