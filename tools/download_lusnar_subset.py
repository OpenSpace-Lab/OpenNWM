#!/usr/bin/env python3
"""Download selected LuSNAR ZIP entries using validated HTTP byte ranges.

Only requests and the Python standard library are required.  Existing files are
reused only after their length and CRC match the archive's central directory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import re
import struct
import sys
import threading
import time
import uuid
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urlencode, urlparse

import requests


REPO = "xuboluo2001/LuSNAR"
BASE = "https://huggingface.co"
CHUNK_LIMIT = 64 * 1024 * 1024
CSTCLOUD_FIDS = {1: "2183307970215941", 2: "2183307970215942", 3: "2183307970215944",
                 4: "2183307970215945", 5: "2183307970215947", 6: "2183307970215943",
                 7: "2183307970215946", 8: "2183307970215948", 9: "2183307970215949"}
PRINT_LOCK = threading.Lock()
LOCAL = threading.local()


def log(message):
    with PRINT_LOCK:
        print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)


def session():
    if not hasattr(LOCAL, "session"):
        LOCAL.session = requests.Session()
        LOCAL.session.headers["User-Agent"] = "LuSNAR-selective-downloader/1.0"
    return LOCAL.session


def direct_session():
    if not hasattr(LOCAL, "direct_session"):
        LOCAL.direct_session = requests.Session()
        LOCAL.direct_session.trust_env = False
        LOCAL.direct_session.headers["User-Agent"] = "LuSNAR-selective-downloader/1.0"
    return LOCAL.direct_session


def request(method, url, http_session=None, **kwargs):
    """Retry transport/transient failures without exposing signed redirect URLs."""
    for attempt in range(6):
        try:
            response = (http_session or session()).request(method, url, timeout=(30, 120), **kwargs)
        except requests.RequestException as error:
            if attempt == 5:
                raise RuntimeError(f"HTTP transport failed ({type(error).__name__})") from None
        else:
            if response.status_code not in (408, 425, 429, 500, 502, 503, 504):
                return response
            status = response.status_code
            response.close()
            if attempt == 5:
                raise RuntimeError(f"HTTP failed after retries: status {status}")
        time.sleep(min(2 ** attempt, 20))
    raise AssertionError("unreachable")


def resolve_url(revision, scene):
    return f"{BASE}/datasets/{REPO}/resolve/{quote(revision, safe='')}/Moon_{scene}.zip"


def cstcloud_url(scene):
    return "https://pan.cstcloud.cn/s/api/shareDownload?" + urlencode(
        {"shareId": "2Ie7D5PSLU", "password": "fjZt", "fid": CSTCLOUD_FIDS[scene]})


def pin_revision(revision, scene):
    if re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        return revision
    with request("HEAD", resolve_url(revision, scene), allow_redirects=False) as response:
        if response.status_code not in (200, 302, 303, 307, 308):
            raise RuntimeError(f"Cannot resolve dataset revision: HTTP {response.status_code}")
        commit = response.headers.get("X-Repo-Commit")
    if not commit or not re.fullmatch(r"[a-fA-F0-9]{40}", commit):
        raise RuntimeError("Resolve response has no immutable X-Repo-Commit; pass --revision COMMIT_SHA")
    return commit


def repository_tree(revision):
    url = f"{BASE}/api/datasets/{REPO}/tree/{quote(revision, safe='')}?recursive=true&expand=false"
    entries = []
    while url:
        with request("GET", url) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Cannot list dataset repository: HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, list):
                raise RuntimeError("Unexpected repository tree response")
            entries.extend(payload)
            url = response.links.get("next", {}).get("url")
    return entries


class RemoteZip(io.RawIOBase):
    """A seekable archive reader that categorically refuses a full-body fallback."""

    def __init__(self, url, size, direct_cdn=False, direct_http=False):
        super().__init__()
        self.url, self.size, self.position = url, size, 0
        self.cached_start, self.cached = 0, b""
        self.direct_cdn = direct_cdn
        self.direct_http = direct_http

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset if whence == io.SEEK_SET else self.position + offset if whence == io.SEEK_CUR else self.size + offset
        if position < 0:
            raise ValueError("Negative archive seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        if end <= self.position:
            return b""
        if not (self.cached_start <= self.position and end <= self.cached_start + len(self.cached)):
            # Metadata reads remain small; zipfile reads the complete central directory.
            self.cached_start = self.position
            self.cached = self.range(self.position, end)
        data = self.cached[self.position - self.cached_start:end - self.cached_start]
        self.position = end
        return data

    def range(self, start, end):
        if not 0 <= start < end <= self.size:
            raise ValueError(f"Invalid archive range {start}:{end}/{self.size}")
        # Separate cache keys prevent proxies reusing one Range response for another.
        url = self.url + ("&" if "?" in self.url else "?") + urlencode({"range_start": start, "range_end": end - 1})
        expected = f"bytes {start}-{end - 1}/{self.size}"
        for attempt in range(6):
            headers = {"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"}
            data_url, data_session = url, direct_session() if self.direct_http else None
            if self.direct_cdn:
                with request("GET", url, headers=headers, stream=True, allow_redirects=False) as resolved:
                    if resolved.status_code not in (302, 303, 307, 308):
                        raise RuntimeError(f"Direct CDN resolve requires a redirect: HTTP {resolved.status_code}")
                    data_url = resolved.headers.get("Location", "")
                    target = urlparse(data_url)
                    if target.scheme != "https" or not target.hostname or not (
                        target.hostname.endswith(".hf.co") or target.hostname.endswith(".huggingface.co")
                    ):
                        raise RuntimeError("Direct CDN resolve returned an unexpected location")
                data_session = direct_session()
            with request("GET", data_url, http_session=data_session, headers=headers, stream=True) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"Server refused range {start}:{end}: HTTP {response.status_code}; full download prohibited")
                if response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"Incorrect Content-Range for {start}:{end}; full download prohibited")
                length = response.headers.get("Content-Length")
                if length is not None and int(length) != end - start:
                    raise RuntimeError("Incorrect range Content-Length")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise RuntimeError("Unexpected HTTP content encoding")
                try:
                    data = bytearray()
                    for block in response.iter_content(1024 * 1024):
                        data.extend(block)
                        if len(data) > end - start:
                            raise RuntimeError("Server returned excess range data")
                except requests.RequestException as error:
                    if attempt == 5:
                        raise RuntimeError(f"Range transfer failed ({type(error).__name__})") from None
                else:
                    if len(data) == end - start:
                        return bytes(data)
                    if attempt == 5:
                        raise RuntimeError(f"Short range transfer: {len(data)} != {end - start}")
            time.sleep(min(2 ** attempt, 20))
        raise AssertionError("unreachable")


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative_path(name, scene):
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts or "\0" in name:
        raise ValueError(f"Unsafe ZIP member path: {name!r}")
    parts = path.parts
    if parts[0].lower() == f"moon_{scene}":
        parts = parts[1:]
    if not parts:
        raise ValueError(f"Empty ZIP member path: {name!r}")
    return str(PurePosixPath(f"Moon_{scene}", *parts))


def modality(name, include_pose, layout):
    parts = PurePosixPath(name.replace("\\", "/").lower()).parts
    camera, folder = layout
    if (PurePosixPath(parts[-1]).suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            and any(parts[index:index + 2] == layout for index in range(len(parts) - 1))):
        return "left_rgb"
    if len(parts) >= 2 and parts[-2:] == (camera, folder + "_timestamp.txt"):
        return "left_timestamps"
    if parts and re.fullmatch(r"imu[^/]*\.txt", parts[-1]):
        return "imu"
    if include_pose and parts and parts[-1] in ("rover_pose.txt", "gt.txt"):
        return "pose"
    return None


def validate_records(records, size):
    if not isinstance(records, list) or not records:
        raise ValueError("Empty or malformed inventory members")
    previous_end = None
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("name"), str):
            raise ValueError("Malformed inventory member")
        name = record["name"]
        path = PurePosixPath(name.replace("\\", "/"))
        if path.is_absolute() or not path.parts or ".." in path.parts or "\0" in name:
            raise ValueError("Unsafe inventory member path")
        for key in ("offset", "end", "compressed_size", "size", "compression", "flags"):
            if type(record.get(key)) is not int or record[key] < 0:
                raise ValueError(f"Invalid inventory field {key}")
        if not 0 <= record["offset"] < record["end"] <= size:
            raise ValueError("Invalid inventory member bounds")
        if previous_end is not None and record["offset"] != previous_end:
            raise ValueError("Noncontiguous or overlapping inventory member bounds")
        if record["compressed_size"] + 30 > record["end"] - record["offset"]:
            raise ValueError("Inventory payload larger than member")
        if not isinstance(record.get("crc32"), str) or not re.fullmatch(r"[0-9a-f]{8}", record["crc32"]):
            raise ValueError("Invalid inventory CRC")
        previous_end = record["end"]


def inventory(scene, size, revision, include_pose, metadata, direct_cdn=False, source="huggingface"):
    remote = RemoteZip(cstcloud_url(scene) if source == "cstcloud" else resolve_url(revision, scene),
                       size, direct_cdn=direct_cdn, direct_http=source == "cstcloud")
    cache_path = metadata / f"Moon_{scene}.inventory.json"
    layouts = (("image0", "color"), ("image1", "rgb"))
    records, layout = None, None
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("revision") == revision and cached.get("archive_size") == size:
                candidate_layout = tuple(cached["left_camera_directory"].split("/"))
                if candidate_layout not in layouts:
                    raise ValueError("Unsupported cached camera layout")
                validate_records(cached["members"], size)
                records, layout = cached["members"], candidate_layout
                log(f"Moon_{scene}: using cached central directory inventory")
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            log(f"Moon_{scene}: invalid inventory cache; reading central directory again")
    if records is None:
        # Always anchor the directory/CRCs to the immutable Hugging Face commit,
        # even when fetching payloads from the original CSTCloud share.
        metadata_remote = remote if source == "huggingface" else RemoteZip(resolve_url(revision, scene), size)
        with zipfile.ZipFile(metadata_remote) as archive:
            infos = sorted(archive.infolist(), key=lambda item: item.header_offset)
            directory_start = archive.start_dir
        records = [{"name": item.filename, "offset": item.header_offset,
                    "end": infos[index + 1].header_offset if index + 1 < len(infos) else directory_start,
                    "compressed_size": item.compress_size, "size": item.file_size,
                    "crc32": f"{item.CRC:08x}", "compression": item.compress_type,
                    "flags": item.flag_bits, "modality": None}
                   for index, item in enumerate(infos)]
        validate_records(records, size)
        layout = next((candidate for candidate in layouts if any(
            "/" + "/".join(candidate) + "/" in "/" + record["name"].lower().replace("\\", "/")
            for record in records if not record["name"].endswith("/"))), None)
        if layout is None:
            raise ValueError(f"Unknown left-camera directory layout in Moon_{scene}")
    selected, seen = [], set()
    for record in records:
        kind = None if record["name"].endswith("/") else modality(record["name"], include_pose, layout)
        record["modality"] = kind
        record.pop("path", None)
        if kind:
            record["path"] = relative_path(record["name"], scene)
            if record["path"] in seen:
                raise ValueError(f"Duplicate output member: {record['path']}")
            seen.add(record["path"])
            if record["flags"] & 1 or record["compression"] not in (0, 8):
                raise ValueError(f"Unsupported encryption or compression: {record['name']}")
            selected.append(record)
    atomic_json(cache_path, {"revision": revision, "archive_size": size, "left_camera_directory": "/".join(layout), "members": records})
    if not any(item["modality"] == "left_rgb" for item in selected):
        raise ValueError(f"No left image files found in Moon_{scene}; inspect archive inventory")
    if not any(item["modality"] == "imu" for item in selected):
        raise ValueError(f"No IMU files found in Moon_{scene}; inspect archive inventory")
    return remote, selected


def valid_existing(path, record):
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    crc = 0
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(data, crc)
    return f"{crc & 0xffffffff:08x}" == record["crc32"]


def pending_members(members, output, workers):
    directory_names, existing = {}, []
    for member in members:
        relative = Path(member["path"])
        parent = relative.parent
        if parent not in directory_names:
            try:
                with os.scandir(output / parent) as entries:
                    directory_names[parent] = {entry.name for entry in entries}
            except FileNotFoundError:
                directory_names[parent] = set()
        if relative.name in directory_names[parent]:
            existing.append(member)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        checks = executor.map(lambda member: valid_existing(output / member["path"], member), existing)
        verified = {member["path"] for member, valid in zip(existing, checks) if valid}
    return [member for member in members if member["path"] not in verified]


def group_members(members):
    groups = []
    for member in members:
        if groups and groups[-1][-1]["end"] == member["offset"] and member["end"] - groups[-1][0]["offset"] <= CHUNK_LIMIT:
            groups[-1].append(member)
        else:
            groups.append([member])
    return groups


def extract_group(remote, members, output):
    start, end = members[0]["offset"], members[-1]["end"]
    data = remote.range(start, end)
    completed = []
    for member in members:
        offset = member["offset"] - start
        if offset + 30 > len(data):
            raise ValueError("Truncated ZIP local header")
        signature, version, flags, compression, mtime, mdate, crc, csize, usize, nlen, xlen = struct.unpack_from("<4s5H3I2H", data, offset)
        if signature != b"PK\x03\x04" or compression != member["compression"] or flags & 1:
            raise ValueError(f"Invalid local header: {member['name']}")
        payload_start = offset + 30 + nlen + xlen
        payload_end = payload_start + member["compressed_size"]
        if payload_end > member["end"] - start or payload_start > payload_end:
            raise ValueError(f"ZIP payload extends beyond member: {member['name']}")
        payload = data[payload_start:payload_end]
        if compression == 8:
            decoder = zlib.decompressobj(-15)
            decoded = decoder.decompress(payload, member["size"] + 1)
            if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
                raise ValueError(f"Invalid deflate stream: {member['name']}")
        else:
            decoded = payload
        if len(decoded) != member["size"] or f"{zlib.crc32(decoded) & 0xffffffff:08x}" != member["crc32"]:
            raise ValueError(f"ZIP length/CRC mismatch: {member['name']}")
        destination = output / member["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Refuse writing through symlinks outside the output root.
        if not destination.resolve().is_relative_to(output.resolve()):
            raise ValueError(f"Output path escapes target directory: {destination}")
        temporary = destination.with_name(destination.name + ".part-" + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as handle:
                handle.write(decoded)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        completed.append(member["path"])
    return {"files": completed, "range_bytes": end - start, "output_bytes": sum(member["size"] for member in members)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", type=int, nargs="+", choices=range(1, 10), default=list(range(1, 10)))
    parser.add_argument("--include-pose", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--source", choices=("huggingface", "cstcloud"), default="huggingface",
                        help="Payload source; both sources use the pinned Hugging Face ZIP inventory and CRCs")
    parser.add_argument("--direct-cdn", action="store_true", help="Resolve through configured proxy, then download signed CDN ranges without proxy")
    args = parser.parse_args()
    if not 1 <= args.workers <= 32:
        parser.error("--workers must be between 1 and 32")
    if args.source == "cstcloud" and args.direct_cdn:
        parser.error("--direct-cdn applies only to --source huggingface; CSTCloud always connects directly")
    output = args.output.expanduser().resolve()
    metadata = output / "_download"
    metadata.mkdir(parents=True, exist_ok=True)
    scenes = sorted(set(args.scenes))
    revision = pin_revision(args.revision, scenes[0])
    log(f"Repository {REPO}, immutable revision {revision}")
    log(f"Payload source: {args.source}")
    tree = repository_tree(revision)
    atomic_json(metadata / "repository_tree.json", {"revision": revision, "entries": tree})
    sizes = {entry["path"]: entry["size"] for entry in tree if entry.get("type") == "file"}
    jobs, plans, all_members = [], [], []
    for scene in scenes:
        filename = f"Moon_{scene}.zip"
        if filename not in sizes:
            raise ValueError(f"Repository archive missing: {filename}")
        log(f"Inspecting {filename} central directory")
        remote, members = inventory(scene, sizes[filename], revision, args.include_pose, metadata,
                                    direct_cdn=args.direct_cdn, source=args.source)
        all_members.extend(members)
        counts = {}
        for member in members:
            counts[member["modality"]] = counts.get(member["modality"], 0) + 1
        pending = members if args.plan_only else pending_members(members, output, args.workers)
        groups = group_members(pending)
        entry = {"scene": scene, "counts": counts, "selected_files": len(members),
                 "selected_bytes": sum(member["size"] for member in members),
                 "selected_compressed_bytes": sum(member["compressed_size"] for member in members),
                 "verified_existing_files": len(members) - len(pending),
                 "pending_range_bytes": sum(group[-1]["end"] - group[0]["offset"] for group in groups),
                 "pending_groups": len(groups)}
        plans.append(entry)
        jobs.extend((scene, remote, group) for group in groups)
        log(f"Moon_{scene}: {counts}; {entry['selected_bytes']:,} selected bytes; {entry['verified_existing_files']} verified existing; {len(groups)} ranges")
    plan = {"repository": REPO, "revision": revision, "include_pose": args.include_pose,
            "source": args.source, "direct_cdn": args.direct_cdn,
            "scenes": plans, "selected_files": len(all_members),
            "selected_bytes": sum(member["size"] for member in all_members),
            "pending_range_bytes": sum(entry["pending_range_bytes"] for entry in plans)}
    atomic_json(metadata / "plan.json", plan)
    log(f"TOTAL: {plan['selected_files']:,} files; {plan['selected_bytes']:,} output bytes; {plan['pending_range_bytes']:,} pending HTTP range bytes")
    if args.plan_only:
        log(f"Plan saved to {metadata / 'plan.json'}; no image/IMU payloads downloaded")
        return
    manifest = dict(plan, status="running", completed_groups=0, downloaded_range_bytes=0, downloaded_output_bytes=0)
    atomic_json(metadata / "manifest.json", manifest)
    started = time.monotonic()
    futures = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(extract_group, remote, members, output): scene for scene, remote, members in jobs}
            with (metadata / "verified_downloads.jsonl").open("a", encoding="utf-8") as receipts:
                try:
                    for future in concurrent.futures.as_completed(futures):
                        result = future.result()
                        scene = futures[future]
                        receipts.write(json.dumps(dict(revision=revision, source=args.source, scene=scene, **result)) + "\n")
                        receipts.flush()
                        manifest["completed_groups"] += 1
                        manifest["downloaded_range_bytes"] += result["range_bytes"]
                        manifest["downloaded_output_bytes"] += result["output_bytes"]
                        atomic_json(metadata / "manifest.json", manifest)
                        elapsed = max(time.monotonic() - started, 0.001)
                        speed = manifest["downloaded_range_bytes"] / elapsed / 1024 ** 2
                        log(f"Group {manifest['completed_groups']}/{len(jobs)} Moon_{scene}: {len(result['files'])} CRC-verified files; {manifest['downloaded_range_bytes']:,}/{plan['pending_range_bytes']:,} bytes ({speed:.2f} MiB/s)")
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        atomic_json(metadata / "manifest.json", manifest)
        raise
    manifest.update(status="complete", elapsed_seconds=round(time.monotonic() - started, 2))
    atomic_json(metadata / "manifest.json", manifest)
    log(f"COMPLETE: {len(all_members):,} CRC-verified files in {output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted; completed files preserved for CRC-verified resume")
        sys.exit(130)
    except Exception as error:
        log(f"FAILED: {type(error).__name__}: {error}")
        sys.exit(1)
