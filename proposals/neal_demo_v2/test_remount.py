"""Local-only remount and spawn-import checks; no model/provider/site calls."""
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def spawned_import_probe(connection):
    """Import the actual worker target without entering its runtime function."""
    try:
        import neal_worker
        import neal_harness
        import neal_bridge
        import neal_limits
        connection.send({"worker": str(Path(neal_worker.__file__).resolve()),
            "target_module": neal_worker.worker_entry.__module__,
            "target_callable": callable(neal_worker.worker_entry),
            "app_name": neal_limits.APP_NAME,
            "torch_imported": "torch" in sys.modules,
            "vllm_imported": "vllm" in sys.modules,
            "model_or_browser_called": False})
    finally:
        connection.close()


class RemountTests(unittest.TestCase):
    def test_packaged_shallow_parent_and_control_do_not_resolve_repository_ancestors(self):
        if importlib.util.find_spec("modal") is None:
            self.skipTest("Run this actual SDK import check with the reviewed Modal1.4.3 interpreter")
        code = r'''
import importlib.util,json,os,pathlib,sys
import modal
source_path=sys.argv[1]
original_resolve=pathlib.Path.resolve
def reject_entry_resolution(self,*args,**kwargs):
    if str(self) in {source_path,"/root/modal_neal.py","/root/neal_control.py"}:
        raise AssertionError("Packaged entry must not resolve shallow __file__ ancestors")
    return original_resolve(self,*args,**kwargs)
pathlib.Path.resolve=reject_entry_resolution
spec=importlib.util.spec_from_file_location("neal_remount_probe",source_path)
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
assert module.PACKAGED_RUNTIME is True
assert str(module.ROOT)=="/opt/weight-space"
if hasattr(module,"cpu_preflight"):
    assert type(module.cpu_preflight).__name__=="Function"
    assert type(module.live_demo).__name__=="Function"
    import neal_worker,neal_lifecycle,weight_space_lab.diagnostic as diagnostic
    manifest=neal_lifecycle.source_manifest(pathlib.Path(diagnostic.__file__).parent,pathlib.Path(neal_worker.__file__).parent)
    bindings=module.executed_source_bindings(manifest)
    assert bindings["proposals/neal_demo_v2/modal_neal.py"]["actual_file_path"]==source_path
    assert bindings["proposals/neal_demo_v2/neal_worker.py"]["actual_file_path"]==neal_worker.__file__
    changed=dict(manifest)
    changed["proposals/neal_demo_v2/modal_neal.py"]="0"*64
    try:module.executed_source_bindings(changed)
    except ValueError:pass
    else:raise AssertionError("Changed executing-parent bytes were accepted")
else:
    assert str(module.RUNS)=="/opt/weight-space/runs/neal-demo-v2"
    bindings={}
print(json.dumps({"root":str(module.ROOT),"provider_calls":0,"executed_source_bindings":bindings}))
'''
        env = dict(os.environ, NEAL_PACKAGED_RUNTIME="1", PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(HERE) + os.pathsep + str(ROOT / "src"))
        for filename in ("modal_neal.py", "neal_control.py"):
            with self.subTest(filename=filename):
                handle, raw_path = tempfile.mkstemp(prefix="neal-v2-shallow-", suffix=".py", dir="/tmp")
                path = Path(raw_path)
                try:
                    with os.fdopen(handle, "wb") as stream:
                        stream.write((HERE / filename).read_bytes())
                    result = subprocess.run([sys.executable, "-c", code, str(path)], env=env,
                        capture_output=True, text=True, timeout=20, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    receipt=json.loads(result.stdout)
                    self.assertEqual(receipt["root"], "/opt/weight-space")
                    if filename == "modal_neal.py":
                        self.assertEqual(len(receipt["executed_source_bindings"]),5)
                finally:
                    path.unlink()

    def test_shallow_root_source_exec_and_actual_spawn_import_target(self):
        # /root/file.py has only two ancestors. This reproduces the exact v1
        # failure depth while using source bytes read from the new draft.
        if importlib.util.find_spec("modal") is not None:
            code = r'''
import json,os,sys,types
source=open(sys.argv[1]).read()
module=types.ModuleType("neal_root_mount_probe")
module.__file__="/root/modal_neal.py"
sys.modules[module.__name__]=module
exec(compile(source,module.__file__,"exec"),module.__dict__)
assert str(module.ROOT)=="/opt/weight-space"
print(json.dumps({"root":str(module.ROOT),"provider_calls":0}))
'''
            env = dict(os.environ, NEAL_PACKAGED_RUNTIME="1", PYTHONDONTWRITEBYTECODE="1",
                PYTHONPATH=str(HERE) + os.pathsep + str(ROOT / "src"))
            result = subprocess.run([sys.executable, "-c", code, str(HERE / "modal_neal.py")],
                env=env, capture_output=True, text=True, timeout=20, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
        context = multiprocessing.get_context("spawn")
        incoming, outgoing = context.Pipe(duplex=False)
        child = context.Process(target=spawned_import_probe, args=(outgoing,))
        child.start()
        outgoing.close()
        try:
            self.assertTrue(incoming.poll(15), "spawned helper import did not return")
            receipt = incoming.recv()
            child.join(5)
            self.assertEqual(child.exitcode, 0)
            self.assertEqual(receipt["worker"], str(HERE / "neal_worker.py"))
            self.assertEqual(receipt["app_name"], "adam-1-cu-neal-private-v2")
            self.assertEqual(receipt["target_module"], "neal_worker")
            self.assertTrue(receipt["target_callable"])
            self.assertFalse(receipt["torch_imported"])
            self.assertFalse(receipt["vllm_imported"])
        finally:
            incoming.close()
            if child.is_alive():
                child.terminate()
                child.join(5)
            if child.is_alive():
                child.kill()
                child.join(5)


if __name__ == "__main__":
    unittest.main()
