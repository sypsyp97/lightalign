"""Contract tests for BIMCV byte ranges, case selection, and HU checks."""

import hashlib
import io
import json
import os
from pathlib import Path
import re
import struct
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from align import bimcv


@pytest.fixture
def remote_zip(monkeypatch):
    def install(method):
        payload = b"NIfTI member contents\x00\xff\x80" * 79
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("preceding.txt", b"not the requested member")
            info = zipfile.ZipInfo("nested/scan.nii.gz")
            info.compress_type = method
            info.extra = struct.pack("<HH", 0xCAFE, 7) + b"payload"
            archive.writestr(info, payload)
            info = archive.getinfo(info.filename)
        row = {
            "zip": "synthetic-volume.zip",
            "name": "scan.nii.gz",
            "offset": info.header_offset,
            "csize": info.compress_size,
            "size": info.file_size,
            "crc": info.CRC,
            "method": method,
        }
        state = {"data": stream.getvalue(), "calls": []}

        class FakeRemote:
            def __init__(self, repo, name):
                assert repo == bimcv.REPO
                assert name == row["zip"]

            def read(self, start, n):
                assert 0 <= start <= len(state["data"])
                assert n >= 0
                state["calls"].append((start, n))
                return state["data"][start:start + n]

        monkeypatch.setattr(bimcv, "Remote", FakeRemote)
        data_start = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
        return row, payload, data_start, state

    return install


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED], ids=["stored", "deflated"])
def test_fetch_real_zip_ranges_and_atomic_write(tmp_path, monkeypatch, remote_zip, method):
    row, payload, data_start, state = remote_zip(method)
    dst = tmp_path / row["name"]
    dst.write_bytes(b"previous complete file")
    replacements = []
    real_replace, real_rename = os.replace, os.rename

    def observe_atomic_move(real_move, src, target, *args, **kwargs):
        if Path(target) == dst:
            assert Path(src) != dst
            assert Path(src).parent.stat().st_dev == dst.parent.stat().st_dev
            assert Path(src).read_bytes() == payload
            assert dst.read_bytes() == b"previous complete file"
            replacements.append((Path(src), Path(target)))
        return real_move(src, target, *args, **kwargs)

    monkeypatch.setattr(os, "replace", lambda src, target, *a, **kw: observe_atomic_move(real_replace, src, target, *a, **kw))
    monkeypatch.setattr(os, "rename", lambda src, target, *a, **kw: observe_atomic_move(real_rename, src, target, *a, **kw))
    bimcv.fetch(row, str(dst))

    assert dst.read_bytes() == payload
    assert (row["offset"], 30) in state["calls"]
    assert (data_start, row["csize"]) in state["calls"]
    assert len(replacements) == 1


@pytest.mark.parametrize("damage", ["signature", "method", "size", "crc"])
def test_fetch_rejects_invalid_member_without_overwriting(tmp_path, remote_zip, damage):
    row, _, _, state = remote_zip(zipfile.ZIP_DEFLATED)
    if damage == "signature":
        offset = row["offset"]
        state["data"] = state["data"][:offset] + b"BAD!" + state["data"][offset + 4:]
    elif damage == "method":
        row["method"] = 99
    elif damage == "size":
        row["size"] += 1
    else:
        row["crc"] ^= 0xFFFFFFFF
    dst = tmp_path / "preserved.nii.gz"
    dst.write_bytes(b"existing valid output")

    with pytest.raises(ValueError):
        bimcv.fetch(row, str(dst))

    assert dst.read_bytes() == b"existing valid output"


@pytest.fixture
def case_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(bimcv, "CT_TERMS", re.compile(r"\bct\b"))
    monkeypatch.setattr(bimcv, "RX_TERMS", re.compile(r"\brx\b"))
    monkeypatch.setattr(bimcv, "SAMPLE_PER_MILLE", 1000)
    monkeypatch.setattr(bimcv, "TEST_DIGIT", 2)
    mapped_labels = sorted({label.strip() for values in bimcv.LABEL_MAP.values() for label in values})
    ctrate_names = [
        "Medical material", "Arterial wall calcification", "Cardiomegaly", "Pericardial effusion",
        "Coronary artery wall calcification", "Hiatal hernia", "Lymphadenopathy", "Emphysema",
        "Atelectasis", "Lung nodule", "Lung opacity", "Pulmonary fibrotic sequela", "Pleural effusion",
        "Mosaic attenuation pattern", "Peribronchial thickening", "Consolidation", "Bronchiectasis",
        "Interlobular septal thickening",
    ]
    names = list(dict.fromkeys([*bimcv.LABEL_MAP, *ctrate_names]))[:18]

    def write(rows, *, labels=None, add_coverage=True):
        out, ctrate_emb = tmp_path / "bimcv", tmp_path / "ctrate_emb"
        (out / "index").mkdir(parents=True, exist_ok=True)
        ctrate_emb.mkdir(exist_ok=True)
        metadata = []
        for i, source in enumerate(rows):
            report_id = source["ReportID"] if "ReportID" in source else f"ses-{i}"
            metadata.append({
                "ReportID": report_id, "Report": f"ct Synthetic report {i}.", "Labels": "[]",
                "path": f"{source['PatientID']}_{report_id}_run-1_bp-chest_ct.nii.gz", **source,
            })
        if add_coverage:
            metadata.append({
                "PatientID": "sub-vocabulary-coverage", "ReportID": "ses-coverage", "Report": "rx Vocabulary only.",
                "Labels": repr(mapped_labels), "path": "sub-vocabulary-coverage_ses-coverage_run-1_bp-chest_ct.nii.gz",
                "Report_en": "Vocabulary only.",
            })
        members = {
            row["path"]: {
                "zip": f"part-{i % 2}.zip", "member": f"nested/{row['path']}", "name": row["path"],
                "offset": 101 + 79 * i, "csize": 23 + i, "size": 41 + i, "crc": 0xACDE0000 + i, "method": 8,
            }
            for i, row in enumerate(metadata)
        }
        pd.DataFrame(metadata).to_csv(out / "index" / "curated_ct_report_path_En.csv", index=False)
        pd.DataFrame(members.values()).to_csv(out / "index" / "members.csv", index=False)
        label_names = names if labels is None else labels
        label_doc = {"labels": label_names, "prompts": [p for label in label_names for p in (f"{label}.", f"No {label}.")]}
        (ctrate_emb / "labels.json").write_text(json.dumps(label_doc, indent=2), encoding="utf-8")
        return out, ctrate_emb, members, label_doc

    return write, mapped_labels, names


def test_build_cases_filters_labels_and_first_volume(case_inputs):
    write, all_labels, _ = case_inputs
    first_label = all_labels[0]
    rows = [
        {
            "PatientID": "sub-one-patient", "ReportID": "ses-1",
            "path": "sub-one-patient_ses-1_run-2_bp-chest_ct.nii.gz",
            "Labels": repr(all_labels), "Report_en": "Discard this volume.",
        },
        {
            "PatientID": "sub-one-patient", "ReportID": "ses-1",
            "path": "sub-one-patient_ses-1_run-1_bp-chest_ct.nii.gz",
            "Labels": repr([f"   {first_label}  "]), "Report_en": "  First English report. \t",
        },
        {"PatientID": "sub-blank", "Report_en": "  \t "},
        {"PatientID": "sub-empty", "Report_en": ""},
        {"PatientID": "sub-nan-report", "Report_en": np.nan},
        {"PatientID": "sub-radiograph", "Report": "rx", "Report_en": "An X-ray report."},
        {"PatientID": "sub-mixed", "Report": "rx ct", "Labels": repr(all_labels), "Report_en": "  CT takes precedence.  "},
        {"PatientID": "sub-neither", "Report": "sin hallazgos", "Labels": "[]", "Report_en": "No modality term."},
        {"PatientID": "sub-nan-labels", "Labels": np.nan, "Report_en": "Labels are missing."},
    ]
    out, ctrate_emb, members, label_doc = write(rows)
    bimcv.build_cases(str(out), str(ctrate_emb))
    cases = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    selected = {
        "sub-one-patient_ses-1_run-1_bp-chest_ct.nii.gz": ("sub-one-patient", "First English report.", {first_label}),
        "sub-mixed_ses-6_run-1_bp-chest_ct.nii.gz": ("sub-mixed", "CT takes precedence.", set(all_labels)),
        "sub-neither_ses-7_run-1_bp-chest_ct.nii.gz": ("sub-neither", "No modality term.", set()),
        "sub-nan-labels_ses-8_run-1_bp-chest_ct.nii.gz": ("sub-nan-labels", "Labels are missing.", set()),
    }
    assert len(cases) == len(selected)
    actual = {case["key"]: case for case in cases}
    assert set(actual) == {path.removesuffix(".nii.gz") for path in selected}
    for path, (patient, report, labels) in selected.items():
        key = path.removesuffix(".nii.gz")
        patient_hash = int(hashlib.sha256(patient.encode()).hexdigest(), 16)
        expected = {
            "id": key, "key": key, "patient": patient,
            "split": "test" if (patient_hash // 1000) % 5 == 2 else "train", "report": report,
            **{column: members[path][column] for column in ("zip", "offset", "csize", "size", "crc", "method")},
            **{label: int(bool(labels & {p.strip() for p in pads})) for label, pads in bimcv.LABEL_MAP.items()},
        }
        assert actual[key] == expected
    assert json.loads((out / "labels.json").read_text()) == label_doc


def test_build_cases_default_sample_per_mille():
    assert bimcv.SAMPLE_PER_MILLE == 240


@pytest.mark.parametrize(
    "path",
    [
        "sub-other_ses-invalid_run-1_bp-chest_ct.nii.gz",
        "sub-invalid_ses-other_run-1_bp-chest_ct.nii.gz",
        "sub-other_ses-other_run-1_bp-chest_ct.nii.gz",
        "scan.nii.gz",
    ],
    ids=["wrong-patient", "wrong-report", "both-wrong", "no-prefix-match"],
)
def test_build_cases_drops_mislinked_rows(case_inputs, path):
    write, _, _ = case_inputs
    rows = [
        {"PatientID": "sub-invalid", "ReportID": "ses-invalid", "path": path, "Report_en": "Report."},
        {"PatientID": "sub-valid", "ReportID": "ses-valid", "Report_en": "Report."},
    ]
    out, ctrate_emb, _, _ = write(rows)
    bimcv.build_cases(str(out), str(ctrate_emb))
    cases = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    assert [case["key"] for case in cases] == ["sub-valid_ses-valid_run-1_bp-chest_ct"]


@pytest.mark.parametrize(
    "report_en",
    ["", "  \t ", np.nan, "Something wrong with the data", "Before. Something wrong with the data. After."],
    ids=["empty", "blank", "nan", "error-text", "embedded-error-text"],
)
def test_build_cases_drops_untranslated_rows(case_inputs, report_en):
    write, _, _ = case_inputs
    rows = [
        {"PatientID": "sub-invalid", "ReportID": "ses-invalid", "Report_en": report_en},
        {"PatientID": "sub-valid", "ReportID": "ses-valid", "Report_en": "something wrong with the data"},
    ]
    out, ctrate_emb, _, _ = write(rows)
    bimcv.build_cases(str(out), str(ctrate_emb))
    cases = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    assert [(case["key"], case["report"]) for case in cases] == [
        ("sub-valid_ses-valid_run-1_bp-chest_ct", "something wrong with the data"),
    ]


@pytest.mark.parametrize(
    "shared_report, unique_report",
    [
        ("ct Shared report.", "ct Shared report. "),
        (np.nan, "ct Unique report."),
        ("ct Shared report.", np.nan),
    ],
    ids=["exact-string", "duplicated-nan", "unique-nan"],
)
def test_build_cases_drops_all_shared_report_rows(case_inputs, shared_report, unique_report):
    write, _, _ = case_inputs
    rows = [
        {"PatientID": "sub-duplicate-a", "Report": shared_report, "Report_en": "Report."},
        {"PatientID": "sub-duplicate-b", "Report": shared_report, "Report_en": "Report."},
        {"PatientID": "sub-valid", "ReportID": "ses-valid", "Report": unique_report, "Report_en": "Report."},
    ]
    out, ctrate_emb, _, _ = write(rows)
    bimcv.build_cases(str(out), str(ctrate_emb))
    cases = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    assert [case["key"] for case in cases] == ["sub-valid_ses-valid_run-1_bp-chest_ct"]


@pytest.mark.parametrize("invalid_field", ["path", "Report_en"])
def test_build_cases_shared_report_counts_otherwise_filtered_rows(case_inputs, invalid_field):
    write, _, _ = case_inputs
    rows = [
        {"PatientID": "sub-duplicate-a", "Report": "ct Shared report.", "Report_en": "Report."},
        {
            "PatientID": "sub-duplicate-b", "Report": "ct Shared report.", "Report_en": "Report.",
            invalid_field: "" if invalid_field == "Report_en" else "unlinked.nii.gz",
        },
        {"PatientID": "sub-valid", "ReportID": "ses-valid", "Report_en": "Report."},
    ]
    out, ctrate_emb, _, _ = write(rows)
    bimcv.build_cases(str(out), str(ctrate_emb))
    cases = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    assert [case["key"] for case in cases] == ["sub-valid_ses-valid_run-1_bp-chest_ct"]


@pytest.mark.parametrize("test_digit", [0, 2, 4])
def test_build_cases_hash_sampling_boundary_and_split(case_inputs, monkeypatch, test_digit):
    write, _, _ = case_inputs
    monkeypatch.setattr(bimcv, "SAMPLE_PER_MILLE", 500)
    monkeypatch.setattr(bimcv, "TEST_DIGIT", test_digit)
    train_digit = (test_digit + 1) % 5
    wanted = {(499, test_digit), (499, train_digit), (500, test_digit), (501, train_digit)}
    patients = {}
    for i in range(100_000):
        patient = f"sub-hashed-patient-{i}"
        h = int(hashlib.sha256(patient.encode()).hexdigest(), 16)
        bucket = (h % 1000, (h // 1000) % 5)
        if bucket in wanted:
            patients.setdefault(bucket, patient)
        if set(patients) == wanted:
            break
    assert set(patients) == wanted
    rows = [{"PatientID": patient, "Report_en": "Report."} for patient in patients.values()]
    out, ctrate_emb, _, _ = write(rows)
    bimcv.build_cases(str(out), str(ctrate_emb))
    cases = [json.loads(line) for line in (out / "cases.jsonl").read_text().splitlines()]
    assert {case["patient"]: case["split"] for case in cases} == {
        patients[(499, test_digit)]: "test", patients[(499, train_digit)]: "train",
    }


def test_build_cases_each_label_alias_and_nan(case_inputs):
    write, all_labels, _ = case_inputs
    rows = [{
        "PatientID": f"sub-label-patient-{i}",
        "Labels": repr([f"   {label}  "]), "Report_en": f"  Report {i}.  ",
    } for i, label in enumerate(all_labels)]
    rows.append({"PatientID": "sub-missing-labels", "ReportID": "ses-missing", "Labels": np.nan, "Report_en": "  No labels.  "})
    out, ctrate_emb, _, _ = write(rows)
    bimcv.build_cases(str(out), str(ctrate_emb))
    cases = {case["key"]: case for case in map(json.loads, (out / "cases.jsonl").read_text().splitlines())}
    assert len(cases) == len(all_labels) + 1
    for i, padchest_label in enumerate(all_labels):
        case = cases[f"sub-label-patient-{i}_ses-{i}_run-1_bp-chest_ct"]
        assert case["report"] == f"Report {i}."
        for label, aliases in bimcv.LABEL_MAP.items():
            assert case[label] == int(padchest_label in {alias.strip() for alias in aliases})
    missing = cases["sub-missing-labels_ses-missing_run-1_bp-chest_ct"]
    assert missing["report"] == "No labels."
    assert all(missing[label] == 0 for label in bimcv.LABEL_MAP)


def test_build_cases_copies_label_document(case_inputs):
    write, _, _ = case_inputs
    rows = [{"PatientID": "sub-patient", "Report_en": "Report."}]
    out, ctrate_emb, _, expected = write(rows)
    bimcv.build_cases(str(out), str(ctrate_emb))
    assert json.loads((out / "labels.json").read_text()) == expected


def test_build_cases_rejects_missing_ctrate_label(case_inputs):
    write, _, names = case_inputs
    missing = next(iter(bimcv.LABEL_MAP))
    rows = [{"PatientID": "sub-patient", "Report_en": "Report."}]
    out, ctrate_emb, _, _ = write(rows, labels=[name for name in names if name != missing])
    with pytest.raises(KeyError):
        bimcv.build_cases(str(out), str(ctrate_emb))


def test_build_cases_rejects_unseen_padchest_label(case_inputs):
    write, all_labels, _ = case_inputs
    rows = [{
        "PatientID": "sub-patient", "Report_en": "Report.",
        "Labels": repr(all_labels[1:]),
    }]
    out, ctrate_emb, _, _ = write(rows, add_coverage=False)
    with pytest.raises(KeyError):
        bimcv.build_cases(str(out), str(ctrate_emb))


@pytest.mark.parametrize(
    "sample, expected_ok",
    [
        (np.linspace(-1100, 300, 101), True),
        (np.linspace(-899, 300, 101), False),
        (np.linspace(-1100, 99, 101), False),
        (np.linspace(-800, 50, 101), False),
        (np.array([-1000, -900, *([-500] * 97), 100, 200]), True),
    ],
    ids=["valid-range", "too-little-air", "too-little-dense-tissue", "both-fail", "inclusive-boundaries"],
)
def test_hu_check_strided_quantiles_and_limits(sample, expected_ok):
    hu = torch.full((97, 101), 1_000_000.0, dtype=torch.float32)
    hu.flatten()[::97] = torch.tensor(sample, dtype=torch.float32)
    quantiles, ok = bimcv.hu_check(hu)
    np.testing.assert_allclose(np.asarray(quantiles), np.quantile(sample, [0.01, 0.5, 0.99]), rtol=0, atol=2e-4)
    assert bool(ok) is expected_ok
