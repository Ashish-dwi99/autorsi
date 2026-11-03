"""Provider protocol tests use fake transports; zero API requests or credentials."""
import json
from dataclasses import replace
from types import SimpleNamespace
import unittest

from framework.adapters import AdapterOperationError, ResourceQuote
from framework.contracts import EditorSpec, RecipeSpec
from framework.editors import OpenAIResponsesEditor, GeminiInteractionsEditor


SCHEMA = {"type": "object", "properties": {"scale": {"type": "number"}},
          "required": ["scale"], "additionalProperties": False}


def validate(parameters):
    if set(parameters) != {"scale"} or type(parameters["scale"]) not in (float, int) or not 0 <= parameters["scale"] <= 1:
        raise ValueError("invalid scale")


def config(provider):
    return SimpleNamespace(editor=EditorSpec(provider, "chosen-editor", credential_env="TEST_KEY"),
                           recipe=RecipeSpec("scale", {"scale": 1.0}),
                           goal={"capability": "test"}, target={"model": "different-target"}, evaluator={"id": "fixture"})


def proposal(**updates):
    return {"decision": "propose", "recipe_id": "scale", "parameters": {"scale": 0.5}, "rationale": "hypothesis", **updates}


def response(cls, text):
    if cls is OpenAIResponsesEditor:
        return {"status": "completed", "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": text}]}]}
    return {"status": "completed", "steps": [
        {"type": "thought", "summary": []},
        {"type": "model_output", "content": [{"type": "text", "text": text}]}]}


class EditorTests(unittest.TestCase):
    def adapter(self, cls, reply=None, **kwargs):
        self.calls = []
        def transport(*args):
            self.calls.append(args)
            return reply if reply is not None else response(cls, json.dumps(proposal()))
        return cls(models={"chosen-editor"}, recipe_id="scale", parameter_schema=SCHEMA,
                   validate_parameters=validate, per_call_quote=ResourceQuote(0.25, 30, 1),
                   transport=transport, credential_lookup=lambda name: "fake-test-credential", **kwargs)

    def test_provider_selection_does_not_replace_target_or_call_a_model_on_construction(self):
        for cls in (OpenAIResponsesEditor, GeminiInteractionsEditor):
            editor = self.adapter(cls)
            self.assertEqual(self.calls, [])
            result = editor.propose(config(editor.provider), {}, ())
            self.assertEqual(result.value.parameters["scale"], 0.5)
            self.assertIsNone(result.usage.cost_usd)
            self.assertEqual(result.usage.model_calls, 1)
            self.assertEqual(len(self.calls), 1)
            endpoint, headers, payload, timeout = self.calls[0]
            self.assertEqual(endpoint, editor.endpoint)
            self.assertEqual(payload["model"], "chosen-editor")
            self.assertIn("different-target", payload["input"])
            self.assertNotIn("fake-test-credential", payload["input"])
            self.assertNotIn("TEST_KEY", payload["input"])
            self.assertIs(payload["store"], False)
            self.assertEqual(timeout, 30)

    def test_provider_native_schema_and_token_limits(self):
        for cls in (OpenAIResponsesEditor, GeminiInteractionsEditor):
            editor = self.adapter(cls, max_output_tokens=123)
            editor.propose(config(editor.provider), {}, ())
            payload = self.calls[0][2]
            if cls is OpenAIResponsesEditor:
                self.assertEqual(payload["max_output_tokens"], 123)
                self.assertTrue(payload["text"]["format"]["strict"])
            else:
                self.assertEqual(payload["generation_config"]["max_output_tokens"], 123)
                self.assertEqual(payload["response_format"]["mime_type"], "application/json")

    def test_stop_is_not_a_success_claim(self):
        editor = self.adapter(OpenAIResponsesEditor, response(OpenAIResponsesEditor, json.dumps(proposal(decision="stop"))))
        self.assertIsNone(editor.propose(config(editor.provider), {}, ()).value)

    def test_malformed_duplicate_and_invented_result_fields_fail(self):
        texts = ["not JSON", '{"decision":"stop","decision":"propose"}',
                 json.dumps(proposal(success=True)), json.dumps(proposal(recipe_id="execute-code")),
                 json.dumps(proposal(parameters={"scale": 4})),
                 json.dumps(proposal(parameters={"scale": True})),
                 json.dumps(proposal(rationale="x" * 4001))]
        for cls in (OpenAIResponsesEditor, GeminiInteractionsEditor):
            for text in texts:
                editor = self.adapter(cls, response(cls, text))
                with self.subTest(cls=cls.__name__, text=text[:40]), self.assertRaises(AdapterOperationError) as caught:
                    editor.propose(config(editor.provider), {}, ())
                self.assertTrue(caught.exception.disposition_known)
                self.assertIsNone(caught.exception.usage.cost_usd)

    def test_incomplete_and_pending_never_become_proposals(self):
        for state, known in (("incomplete", True), ("in_progress", False), ("requires_continuation", False)):
            editor = self.adapter(GeminiInteractionsEditor, {"status": state})
            with self.assertRaises(AdapterOperationError) as caught:
                editor.propose(config(editor.provider), {}, ())
            self.assertEqual(caught.exception.disposition_known, known)
            self.assertEqual(len(self.calls), 1)

    def test_refusal_and_tool_calls_are_not_executed(self):
        for cls in (OpenAIResponsesEditor, GeminiInteractionsEditor):
            reply = response(cls, "{}")
            if cls is OpenAIResponsesEditor:
                reply["output"][-1]["content"] = [{"type": "refusal", "refusal": "No"}]
            else:
                reply["steps"].append({"type": "function_call", "name": "shell"})
            editor = self.adapter(cls, reply)
            with self.assertRaises(AdapterOperationError):
                editor.propose(config(editor.provider), {}, ())

    def test_missing_credential_stops_before_network(self):
        editor = self.adapter(OpenAIResponsesEditor)
        editor.credential_lookup = lambda name: None
        with self.assertRaises(AdapterOperationError) as caught:
            editor.propose(config(editor.provider), {}, ())
        self.assertEqual(self.calls, [])
        self.assertEqual(caught.exception.usage.model_calls, 0)

    def test_timeout_is_unknown_without_retry_and_does_not_leak_exception(self):
        editor = self.adapter(OpenAIResponsesEditor)
        def fail(*args):
            self.calls.append(args)
            raise TimeoutError("sensitive-transport-details")
        editor.transport = fail
        with self.assertRaises(AdapterOperationError) as caught:
            editor.propose(config(editor.provider), {}, ())
        self.assertFalse(caught.exception.disposition_known)
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("sensitive", str(caught.exception))
        self.assertIsNone(caught.exception.usage.model_calls)

    def test_endpoint_and_model_mismatch_fail_before_network(self):
        editor = self.adapter(OpenAIResponsesEditor)
        for spec in (EditorSpec("openai", "unregistered", credential_env="TEST_KEY"),
                     EditorSpec("gemini", "chosen-editor", credential_env="TEST_KEY"),
                     EditorSpec("openai", "chosen-editor", "https://different.example", "TEST_KEY")):
            value = config(editor.provider)
            value.editor = spec
            with self.assertRaises(ValueError):
                editor.propose(value, {}, ())
        self.assertEqual(self.calls, [])

    def test_no_unbounded_parameter_object(self):
        with self.assertRaises(ValueError):
            OpenAIResponsesEditor(models={"chosen-editor"}, recipe_id="scale", parameter_schema={"type": "object"},
                                  validate_parameters=validate, per_call_quote=ResourceQuote(1, 30, 1))


if __name__ == "__main__":
    unittest.main()
