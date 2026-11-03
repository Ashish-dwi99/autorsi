"""Fixed independent verification: original plus one previously selected edited candidate."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from weight_space_lab.benchmarks.miniwob import Case, FAMILIES
from action_schema import schema_receipt

VERIFICATION_SEED = 2026110397
FIXED_CANDIDATE = 0.5
ALLOWED_CANDIDATES = (FIXED_CANDIDATE,)
SCALES = (1.0, *ALLOWED_CANDIDATES)  # Bounds only; executable pair comes from frozen selection.
DEVELOPMENT_SOURCE_SHA256 = "b44dbbf430face18a694a4a8a3f4f6d8e10f1abf841806a2b9fba5ef401fde85"
APP_NAME = "adam-1-verify-miniwob-20261104-v1"
IMPORT_RUN = "adam-verify-imports-20261104-v1"
GPU_RUN = "adam-verify-miniwob-20261104-v1"
CHILD_WALL_SECONDS = 1200
NEW_WORK_SECONDS = 1140
GENERATION_HARD_SECONDS = 45
MAX_MODEL_REQUESTS = 194  # Two neutral warmups plus at most24*8 scored requests.
MAX_IMAGE_BYTES = 128 * 1024
MAX_IMAGES = 192
MAX_ARTIFACT_BYTES = MAX_IMAGES * MAX_IMAGE_BYTES
NOT_AFTER_UNIX = 1793880000  # 2026-11-05 12:00UTC; no automatic renewal.
PAIRING_RULE = "fixed12families; alternate original/candidate order,6cases each; no reselection"
FILES = ("verify_contract.py", "verify_patch.py", "verify_native.py", "verify_benchmark.py",
         "verify_entry.py", "modal_verify.py", "verify_host.py", "build_miniwob_runtime.py", "action_schema.py", "development_contract.py", "SCOPE.json")


def cases():
    return tuple(Case(family, VERIFICATION_SEED) for family in FAMILIES)


def scales(candidate):
    if type(candidate) not in (float,int) or candidate not in ALLOWED_CANDIDATES:
        raise ValueError("the predeclared edited candidate .5 is required")
    return (1.0,float(candidate))


def variant_order(index,candidate):
    if type(index) is not int or not 0<=index<12:
        raise ValueError("verification case index required")
    pair=scales(candidate)
    return pair if index%2==0 else pair[::-1]


def reset_variant_caches(engine):
    """Same cold image/text cache state for each paired variant under native clocks."""
    if engine.reset_prefix_cache() is not True:
        raise RuntimeError("text prefix cache reset did not confirm success; no inference permitted")
    engine.reset_mm_cache()
    engine.llm_engine.reset_encoder_cache()
    return {"prefix_reset_returned_true": True, "mm_reset_returned": True,
            "encoder_reset_returned": True, "purpose": "Equal fresh image/text cache state per variant"}


def experiment_spec(candidate):
    return {"phase":"verification","verification_seed":VERIFICATION_SEED,
        "scales":list(scales(candidate)),"case_ids":[case.case_id for case in cases()],
        "pairing_rule":PAIRING_RULE,"candidate_reselection":False,
        "max_model_requests_including_warmups":MAX_MODEL_REQUESTS,
        "protocol":"action-only-json-v1-with-structural-grammar-v2",
        "structural_decoding":schema_receipt(),"max_completion_tokens":96,
        "actions_per_episode":8,"native_timers_unchanged":True}


def validate_candidate(admission):
    """Completed development + separate root quality/selection before any reset."""
    from development_contract import select_development,experiment_spec as development_spec
    from action_schema import validate_cpu_grammar
    report=admission.get('development_report',{})
    child=report.get('child',{})
    source_hash=hashlib.sha256(json.dumps(report.get('source_manifest'),sort_keys=True,allow_nan=False).encode()).hexdigest()
    if (report.get('run')!='adam-schema-miniwob-20261104-v2' or report.get('status')!='completed'
        or report.get('child_reaped') is not True or report.get('child_exitcode')!=0
        or source_hash!=DEVELOPMENT_SOURCE_SHA256
        or child.get('status')!='completed_development_only' or child.get('complete_coverage') is not True
        or child.get('browser_close_returned') is not True or child.get('verification_performed') is not False
        or child.get('experiment_spec')!=development_spec() or child.get('structural_decoding')!=schema_receipt()):
        raise ValueError('completed exact corrected development result required')
    validate_cpu_grammar(child.get('grammar_preflight',{}))
    selected=select_development(child.get('variants',{}))
    if selected!=child.get('development_selection'):
        raise ValueError('development selection does not recompute')
    candidate=scales(selected['selected_scale'])[1]
    selection=admission.get('candidate_selection',{})
    report_hash=admission.get('development_report_sha256')
    if (selection.get('schema')!=1 or selection.get('purpose')!='reserved-miniwob-verification'
        or selection.get('development_report_sha256')!=report_hash
        or selection.get('source_manifest_sha256')!=DEVELOPMENT_SOURCE_SHA256
        or selection.get('selected_attention_scale')!=candidate
        or selection.get('development_selection')!=selected
        or selection.get('protocol_quality_reviewed') is not True
        or selection.get('verification_previously_performed') is not False
        or selection.get('quality_review_sha256')!=admission.get('quality_review_sha256')
        or not isinstance(selection.get('quality_review_path'),str)
        or not selection['quality_review_path'].startswith('evidence/')):
        raise ValueError('separately frozen candidate selection required')
    quality=admission.get('quality_review',{})
    if (quality.get('schema')!=1 or quality.get('development_report_sha256')!=report_hash
        or quality.get('source_manifest_sha256')!=DEVELOPMENT_SOURCE_SHA256
        or quality.get('complete_episodes')!=36
        or any(type(quality.get(k)) is not int or quality[k]!=0
            for k in ('malformed_outputs','missing_envelope_outputs','unknown_operations'))
        or quality.get('semantic_invalid_actions_reviewed') is not True
        or quality.get('carry_forward_to_miniwob_verification') is not True
        or quality.get('promotion')!='NONE'):
        raise ValueError('separate satisfactory root protocol review required')
    return candidate


def validate_spec(admission):
    candidate=validate_candidate(admission)
    if admission.get('experiment_spec')!=experiment_spec(candidate):
        raise ValueError('exact fixed independent verification pair required')
    return candidate


def compare_verification(variants,candidate):
    """Report all24 outcomes; never select a new recipe or authorize a retry."""
    pair=scales(candidate)
    if set(variants)!={str(value) for value in pair}:
        raise ValueError('exact original and fixed candidate required')
    expected=[case.case_id for case in cases()]
    scores=[]
    for value in pair:
        rows=variants[str(value)]
        if len(rows)!=12 or [r.get('case_id') for r in rows]!=expected:
            raise ValueError('all24 reserved paired outcomes required')
        if any(type(r.get('success')) is not bool or type(r.get('invalid_actions')) is not int
            or r['invalid_actions']<0 or type(r.get('completion_tokens')) is not int
            or r['completion_tokens']<0 or r.get('token_usage_complete') is not True for r in rows):
            raise ValueError('complete actual outcomes/tokens required')
        scores.append({'scale':value,'successes':sum(r['success'] for r in rows),
            'invalid_actions':sum(r['invalid_actions'] for r in rows),
            'completion_tokens':sum(r['completion_tokens'] for r in rows),'denominator':12})
    baseline,edited=variants['1.0'],variants[str(candidate)]
    return {'fixed_candidate_scale':candidate,'scores':scores,
        'paired_gains':[a['case_id'] for a,b in zip(baseline,edited) if not a['success'] and b['success']],
        'paired_regressions':[a['case_id'] for a,b in zip(baseline,edited) if a['success'] and not b['success']],
        'candidate_perfect_on_this_subset':all(r['success'] for r in edited),
        'candidate_reselected':False,'retry_authorized':False,'promotion':'NONE; independent result only',
        'scope':'Twelve reserved seeded DOM-assisted MiniWoB tasks; not universal or Astra evidence'}


def source_manifest(package_root, proposal_root):
    package_root, proposal_root = Path(package_root), Path(proposal_root)
    paths = {"proposals/adam_verify_v1/" + name: proposal_root / name for name in FILES}
    for path in sorted(package_root.rglob("*")):
        rel = path.relative_to(package_root)
        if "__pycache__" in rel.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError("source symlink refused")
        if path.is_file():
            if path.suffix not in {".py", ".json", ".txt", ".jinja", ".md"}:
                raise ValueError("unexpected package source type")
            paths["src/weight_space_lab/" + str(rel)] = path
    if len(paths) > 80:
        raise ValueError("source inventory too large")
    output, total = {}, 0
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing ordinary source: " + name)
        raw = path.read_bytes(); total += len(raw)
        if total > 2 * 1024**2:
            raise ValueError("source bytes exceed2MiB")
        output[name] = hashlib.sha256(raw).hexdigest()
    return output


def executing_source_join(manifest, modules):
    """Join SDK auto-mounted modules to explicit packaged copies, by actual bytes."""
    modules = {"verify_contract.py": __file__, **modules}
    output = {}
    for name, filename in modules.items():
        expected = manifest["proposals/adam_verify_v1/" + name]
        path = Path(filename)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("executing source differs from packaged manifest: " + name)
        output[name] = {"actual_path": str(path), "sha256": actual}
    return output


def save_observation(observation, destination, *, case_id, scale, dispatch, elapsed_s):
    """Exact returned pre-action screenshot, not a reconstruction or continuous video."""
    import io
    from PIL import Image
    if type(dispatch) is not int or not 1 <= dispatch <= 8 or scale not in SCALES:
        raise ValueError("unexpected observation identity")
    if not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("invalid original elapsed time")
    image = Image.fromarray(observation["screenshot"])
    if image.size != (160, 210):
        raise ValueError("unchanged MiniWoB screenshot geometry required")
    data = io.BytesIO(); image.save(data, format="PNG"); raw = data.getvalue()
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("observation image exceeds128KiB")
    path = Path(destination)
    with path.open("xb") as stream:
        stream.write(raw)
    return {"relative_path": path.name, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "case_id": case_id, "scale": scale, "dispatch": dispatch, "elapsed_s": elapsed_s,
            "width":160,"height":210,"kind":"actual_pre_action_observation"}
