#!/usr/bin/env python3

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request


REPO_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

DEFAULT_MANIFEST = (
    REPO_ROOT
    / "artifact_manifest.json"
)

DEFAULT_ARTIFACT_ROOT = (
    REPO_ROOT
    / "artifacts"
)


CHUNK_SIZE = (
    8
    *
    1024
    *
    1024
)


def sha256_file(
    path,
    block_size=16 * 1024 * 1024,
):

    digest = hashlib.sha256()


    with Path(path).open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                block_size
            )


            if not block:

                break


            digest.update(
                block
            )


    return digest.hexdigest()


def prepare_partial(
    target,
    expected_size,
):

    target = Path(
        target
    )

    part = Path(
        str(
            target
        )
        +
        ".part"
    )


    # A previous urllib.urlretrieve failure may have left
    # a partial file directly at target.
    if target.exists():

        size = target.stat().st_size


        if size == expected_size:

            return part


        if (
            0
            <
            size
            <
            expected_size
        ):

            # Keep whichever partial copy is larger.
            if part.exists():

                if (
                    part.stat().st_size
                    >=
                    size
                ):

                    target.unlink()

                else:

                    part.unlink()

                    target.replace(
                        part
                    )

            else:

                target.replace(
                    part
                )


        else:

            target.unlink()


    if (
        part.exists()
        and
        part.stat().st_size
        >
        expected_size
    ):

        part.unlink()


    return part


def resumable_download(
    *,
    url,
    target,
    expected_size,
    max_retries=10,
):

    target = Path(
        target
    )

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    if (
        target.exists()
        and
        target.stat().st_size
        ==
        expected_size
    ):

        return {
            "downloaded":
                False,

            "resumed":
                False,

            "retries":
                0,
        }


    part = prepare_partial(
        target,
        expected_size,
    )


    ever_resumed = (
        part.exists()
        and
        part.stat().st_size
        >
        0
    )


    retries_used = 0


    for attempt in range(
        1,
        max_retries
        +
        1,
    ):

        current = (
            part.stat().st_size
            if part.exists()
            else 0
        )


        if current == expected_size:

            os.replace(
                part,
                target
            )

            return {
                "downloaded":
                    True,

                "resumed":
                    ever_resumed,

                "retries":
                    retries_used,
            }


        if current > expected_size:

            part.unlink(
                missing_ok=True
            )

            current = 0


        headers = {
            "User-Agent":
                "VietLegal-RAG-artifact-downloader/1.1",

            "Accept-Encoding":
                "identity",
        }


        if current > 0:

            headers[
                "Range"
            ] = (
                f"bytes={current}-"
            )


        request = urllib.request.Request(
            url,
            headers=headers,
        )


        try:

            with urllib.request.urlopen(
                request,
                timeout=180,
            ) as response:

                status = getattr(
                    response,
                    "status",
                    200,
                )


                # If Range was requested but ignored, restart safely.
                if (
                    current > 0
                    and
                    status != 206
                ):

                    mode = "wb"

                    current = 0

                else:

                    mode = (
                        "ab"
                        if current > 0
                        else "wb"
                    )


                    if current > 0:

                        ever_resumed = True


                with part.open(
                    mode
                ) as f:

                    while True:

                        chunk = response.read(
                            CHUNK_SIZE
                        )


                        if not chunk:

                            break


                        f.write(
                            chunk
                        )


                    f.flush()

                    os.fsync(
                        f.fileno()
                    )


            final_size = (
                part.stat().st_size
            )


            if final_size == expected_size:

                os.replace(
                    part,
                    target
                )

                return {
                    "downloaded":
                        True,

                    "resumed":
                        ever_resumed,

                    "retries":
                        retries_used,
                }


            if final_size > expected_size:

                part.unlink(
                    missing_ok=True
                )


                raise IOError(
                    f"download exceeded expected size: "
                    f"{final_size} > {expected_size}"
                )


            # Connection closed cleanly but early.
            raise IOError(
                f"incomplete response: "
                f"{final_size}/{expected_size} bytes"
            )


        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ConnectionError,
            OSError,
            IOError,
        ) as exc:

            retries_used += 1


            current = (
                part.stat().st_size
                if part.exists()
                else 0
            )


            if attempt >= max_retries:

                raise RuntimeError(
                    f"Download failed after "
                    f"{max_retries} attempts. "
                    f"Partial={current}/{expected_size}. "
                    f"URL={url}. "
                    f"Last error={exc}"
                ) from exc


            sleep_seconds = min(
                2 ** (
                    attempt
                    -
                    1
                ),
                30,
            )


            print(
                f"    ⚠ retry {attempt}/{max_retries} "
                f"| partial={current}/{expected_size} "
                f"| sleep={sleep_seconds}s "
                f"| {type(exc).__name__}: {exc}",
                flush=True,
            )


            time.sleep(
                sleep_seconds
            )


    raise RuntimeError(
        "Unexpected downloader state."
    )


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Deploy VietLegal-RAG runtime artifacts."
        )
    )


    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )


    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=DEFAULT_ARTIFACT_ROOT,
    )


    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
    )


    parser.add_argument(
        "--verify-only",
        action="store_true",
    )


    parser.add_argument(
        "--skip-hash",
        action="store_true",
    )


    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
    )


    args = parser.parse_args()


    manifest = json.loads(
        args.manifest.read_text(
            encoding="utf-8"
        )
    )


    artifact_root = (
        args.artifact_root
        .expanduser()
        .resolve()
    )


    artifact_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    source_root = (
        args.source_root
        .expanduser()
        .resolve()

        if args.source_root

        else None
    )


    failures = []

    resumed_count = 0

    downloaded_count = 0

    retry_count = 0


    for i, entry in enumerate(
        manifest[
            "files"
        ],
        start=1,
    ):

        relative = Path(
            entry[
                "path"
            ]
        )


        target = (
            artifact_root
            / relative
        )


        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        expected_size = int(
            entry[
                "size_bytes"
            ]
        )


        # -----------------------------------------------------------------------------------------
        # Acquire file
        # -----------------------------------------------------------------------------------------

        if not args.verify_only:

            if source_root is not None:

                valid_existing = (
                    target.exists()
                    and
                    target.stat().st_size
                    ==
                    expected_size
                )


                if not valid_existing:

                    source = (
                        source_root
                        / relative
                    )


                    if not source.exists():

                        failures.append(
                            (
                                str(
                                    relative
                                ),
                                "missing from source root",
                            )
                        )

                        continue


                    shutil.copy2(
                        source,
                        target,
                    )


            else:

                url = entry.get(
                    "url"
                )


                if not url:

                    failures.append(
                        (
                            str(
                                relative
                            ),
                            "no remote URL configured",
                        )
                    )

                    continue


                try:

                    stats = resumable_download(
                        url=url,

                        target=target,

                        expected_size=
                            expected_size,

                        max_retries=
                            args.max_retries,
                    )


                    if stats[
                        "downloaded"
                    ]:

                        downloaded_count += 1


                    if stats[
                        "resumed"
                    ]:

                        resumed_count += 1


                    retry_count += int(
                        stats[
                            "retries"
                        ]
                    )


                except Exception as exc:

                    failures.append(
                        (
                            str(
                                relative
                            ),
                            str(
                                exc
                            ),
                        )
                    )

                    continue


        # -----------------------------------------------------------------------------------------
        # Size verification
        # -----------------------------------------------------------------------------------------

        if not target.exists():

            failures.append(
                (
                    str(
                        relative
                    ),
                    "missing",
                )
            )

            continue


        if (
            target.stat().st_size
            !=
            expected_size
        ):

            failures.append(
                (
                    str(
                        relative
                    ),
                    (
                        "size mismatch: "
                        f"{target.stat().st_size} "
                        f"!= {expected_size}"
                    ),
                )
            )

            continue


        # -----------------------------------------------------------------------------------------
        # SHA256
        # -----------------------------------------------------------------------------------------

        if not args.skip_hash:

            actual_hash = sha256_file(
                target
            )


            if (
                actual_hash
                !=
                entry[
                    "sha256"
                ]
            ):

                # Do not leave a corrupt apparently complete file.
                target.unlink(
                    missing_ok=True
                )


                failures.append(
                    (
                        str(
                            relative
                        ),
                        "sha256 mismatch",
                    )
                )

                continue


        print(
            f"[{i:03d}/{len(manifest['files']):03d}] "
            f"OK {relative}",
            flush=True,
        )


    if failures:

        print(
            "\nARTIFACT_VERIFICATION_FAILED",
            file=sys.stderr,
        )


        for path, reason in failures:

            print(
                f"- {path}: {reason}",
                file=sys.stderr,
            )


        raise SystemExit(
            1
        )


    # Ensure no stale partial files remain after success.
    stale_parts = list(
        artifact_root.rglob(
            "*.part"
        )
    )


    if stale_parts:

        print(
            "\nUnexpected stale partial files:",
            file=sys.stderr,
        )


        for path in stale_parts:

            print(
                path,
                file=sys.stderr,
            )


        raise SystemExit(
            1
        )


    print(
        "\nARTIFACTS_READY"
    )

    print(
        artifact_root
    )

    print(
        f"downloaded_this_run={downloaded_count}"
    )

    print(
        f"resumed_files={resumed_count}"
    )

    print(
        f"network_retries={retry_count}"
    )


if __name__ == "__main__":

    main()
