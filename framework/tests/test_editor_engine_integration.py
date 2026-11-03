"""Provider adapter/engine integration with injected fake HTTP only."""
from dataclasses import replace
import json
import unittest

from framework.adapters import ResourceQuote
from framework.contracts import EditorSpec
from framework.editors import OpenAIResponsesEditor, GeminiInteractionsEditor
from framework.engine import run_experiment
from framework.tests.test_engine import config, registry


SCHEMA = {"type": "object", "properties": {"strength": {"type": "number", "minimum": 0, "maximum": 1}},
          "required": ["strength"], "additionalProperties": False}


class EditorEngineIntegrationTests(unittest.TestCase):
    def setup_editor(self, cls, transport):
        reg, _, target, recipe, evaluator = registry([])
        adapter = cls(models={"declared-test-editor"}, recipe_id="bounded_strength", parameter_schema=SCHEMA,
            validate_parameters=recipe.validate_parameters, per_call_quote=ResourceQuote(.2, 10, 1),
            transport=transport, credential_lookup=lambda name: "FAKE_SECRET_NOT_REAL")
        reg.register_editor(adapter)
        cfg = replace(config(), editor=EditorSpec(adapter.provider, "declared-test-editor", credential_env="FAKE_EDITOR_KEY"))
        return reg, cfg, target

    def response_text(self, parameters=None, decision="propose"):
        return json.dumps({"decision": decision, "recipe_id": "bounded_strength",
                           "parameters": parameters or {"strength": 1.0}, "rationale": "Test hypothesis only"})

    def test_openai_one_structured_proposal_drives_measured_acceptance(self):
        calls = []
        def transport(endpoint, headers, payload, timeout):
            calls.append(payload)
            self.assertEqual(endpoint, "https://api.openai.com/v1/responses")
            self.assertFalse(payload["store"])
            self.assertTrue(payload["text"]["format"]["strict"])
            return {"status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": self.response_text()}]}]}
        reg, cfg, target = self.setup_editor(OpenAIResponsesEditor, transport)
        result = run_experiment(cfg, reg)
        self.assertTrue(result.succeeded)
        self.assertEqual((len(calls), target.accepts), (1, 1))
        self.assertEqual(result.resources["editor_cost_charged_or_reserved_usd"], .2)
        self.assertNotIn("FAKE_SECRET_NOT_REAL", json.dumps(result.to_dict()))

    def test_gemini_stop_does_not_become_goal_success(self):
        calls = []
        def transport(endpoint, headers, payload, timeout):
            calls.append(payload)
            self.assertEqual(endpoint, "https://generativelanguage.googleapis.com/v1beta/interactions")
            self.assertEqual(payload["response_format"]["mime_type"], "application/json")
            return {"status": "completed", "steps": [{"type": "model_output", "content": [
                {"type": "text", "text": self.response_text(decision="stop")}]}]}
        reg, cfg, target = self.setup_editor(GeminiInteractionsEditor, transport)
        result = run_experiment(cfg, reg)
        self.assertEqual(result.status, "editor_stopped")
        self.assertFalse(result.succeeded)
        self.assertEqual((len(calls), target.accepts, target.state), (1, 0, 0.0))

    def test_unknown_transport_is_not_retried_and_secret_is_sanitized(self):
        calls = []
        def transport(*args):
            calls.append(True)
            raise TimeoutError("FAKE_SECRET_NOT_REAL in a hypothetical provider error")
        reg, cfg, target = self.setup_editor(OpenAIResponsesEditor, transport)
        result = run_experiment(cfg, reg)
        self.assertEqual((len(calls), target.state, target.restores), (1, 0.0, 0))
        self.assertEqual(result.resources["editor_cost_charged_or_reserved_usd"], .2)
        self.assertNotIn("FAKE_SECRET_NOT_REAL", json.dumps(result.to_dict()))

    def test_editor_cannot_add_a_claimed_success_parameter(self):
        def transport(*args):
            return {"status": "completed", "output": [{"type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": self.response_text({"strength": 1.0, "claimed_success": True})}]}]}
        reg, cfg, target = self.setup_editor(OpenAIResponsesEditor, transport)
        result = run_experiment(cfg, reg)
        self.assertEqual(result.status, "candidate_failed")
        self.assertFalse(result.succeeded)
        self.assertEqual((target.state, target.accepts), (0.0, 0))


if __name__ == "__main__":
    unittest.main()
