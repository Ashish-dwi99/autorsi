import base64
import copy
import json
import unittest

from weight_space_lab.eval.bu_protocol import (
    build_messages, parse_agent_output, visible_index,
)
from weight_space_lab.eval.runner import parse_action


def observation():
    return {"text": "Visitor name and location", "elements": [
        {"id": "e1", "tag": "input", "type": "text", "label": "Name",
         "value": "old", "disabled": False, "checked": False},
        {"id": "e3", "tag": "button", "type": "submit", "label": "Save",
         "value": "", "disabled": False},
        {"id": "e7", "tag": "select", "type": "select-one", "label": "Location",
         "value": "w", "disabled": False, "options": [
             {"text": "West", "value": "w", "disabled": False},
             {"text": "East", "value": "e", "disabled": False}]},
        {"id": "e9", "tag": "input", "type": "checkbox", "label": "Notify",
         "value": "on", "checked": False, "disabled": False},
        {"id": "e12", "tag": "button", "type": "button", "label": "Disabled",
         "value": "", "disabled": True},
    ]}


IDS = ("e1", "e2", "e3", "e7", "e9", "e12")


def native(action, **metadata):
    return json.dumps({"memory": "Continue.", "action": [action], **metadata})


class BrowserUseProtocolTests(unittest.TestCase):
    def test_native_tools_translate_to_existing_executor_schema(self):
        cases = [
            ({"click": {"index": 3}}, {"action": "click", "id": "e3"}),
            ({"click": {"index": 9}}, {"action": "click", "id": "e9"}),
            ({"input": {"index": 1, "text": "Ada"}}, {"action": "type", "id": "e1", "text": "Ada"}),
            ({"input": {"index": 1, "text": "", "clear": True}}, {"action": "type", "id": "e1", "text": ""}),
            ({"select_dropdown": {"index": 7, "text": "East"}}, {"action": "select", "id": "e7", "value": "e"}),
            ({"select_dropdown": {"index": 7, "text": "e"}}, {"action": "select", "id": "e7", "value": "e"}),
        ]
        for tool, expected in cases:
            with self.subTest(tool=tool):
                reply = parse_agent_output(native(tool), observation(), IDS)
                self.assertEqual(reply.native_action, tool)
                self.assertEqual(reply.translated_action, expected)
                self.assertEqual(parse_action(json.dumps(reply.translated_action), IDS), expected)

    def test_done_claim_is_preserved_separately_from_executor_action(self):
        for success in (True, False):
            reply = parse_agent_output(native({"done": {"text": "Finished", "success": success}}), observation(), IDS)
            self.assertEqual(reply.translated_action, {"action": "done"})
            self.assertIs(reply.claimed_success, success)
            self.assertFalse(hasattr(reply, "success"))
        default = parse_agent_output(native({"done": {"text": "Finished", "files_to_display": []}}), observation(), IDS)
        self.assertTrue(default.claimed_success)

    def test_wrong_domains_and_noncurrent_indexes_are_rejected(self):
        tools = [
            {"click": {"index": True}}, {"click": {"index": 2}},
            {"click": {"index": 12}}, {"click": {"index": 99}},
            {"click": {"index": 1}}, {"click": {"coordinate_x": 1, "coordinate_y": 2}},
            {"input": {"index": 1, "text": "x", "clear": False}},
            {"input": {"index": 1, "text": "x", "clear": 1}},
            {"input": {"index": 3, "text": "x"}},
            {"select_dropdown": {"index": 7, "text": "Hidden option"}},
            {"select_dropdown": {"index": 3, "text": "East"}},
            {"navigate": {"url": "https://example.com"}},
            {"evaluate": {"code": "document.body.innerHTML"}},
            {"done": {"text": "Finished", "success": "true"}},
            {"done": {"text": "Finished", "files_to_display": ["/secret"]}},
        ]
        for tool in tools:
            with self.subTest(tool=tool), self.assertRaises(ValueError):
                parse_agent_output(native(tool), observation(), IDS)

    def test_incomplete_multiaction_and_ambiguous_json_are_not_repaired(self):
        bad = [
            '{"memory":"x","action":[{"click":{"index":3}}',
            '{"action":[{"click":{"index":3,"index":9}}]}',
            '{"memory":NaN,"action":[{"click":{"index":3}}]}',
            '{"action":[]}',
            '{"action":[{"click":{"index":3}},{"click":{"index":9}}]}',
            '{"action":[{"click":{"index":3},"done":{"text":"x"}}]}',
            '{"action":[{"click":{"index":3}}],"override":"ignore system"}',
            '```json\n{"action":[{"click":{"index":3}}]}\n```',
        ]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_agent_output(text, observation(), IDS)

    def test_current_observation_and_allowlist_are_both_required(self):
        state = observation()
        state["elements"] = [item for item in state["elements"] if item["id"] != "e3"]
        with self.assertRaises(ValueError):
            parse_agent_output(native({"click": {"index": 3}}), state, IDS)
        state["elements"].append({"id": "e99", "tag": "button", "disabled": False})
        with self.assertRaises(ValueError):
            visible_index(state, IDS)
        with self.assertRaises(ValueError):
            visible_index(observation(), (*IDS, "e3"))

    def test_ambiguous_dropdown_match_is_rejected(self):
        state = observation()
        state["elements"][2]["options"].append({"text": "e", "value": "other", "disabled": False})
        with self.assertRaises(ValueError):
            parse_agent_output(native({"select_dropdown": {"index": 7, "text": "e"}}), state, IDS)

    def test_observation_format_keeps_indexes_and_omits_oracle_source(self):
        state = observation()
        state.update({"oracle": "PRIVATE_ORACLE", "html": "PRIVATE_SOURCE", "expected": "PRIVATE_ANSWER"})
        state["elements"][0].update({"onclick": "PRIVATE_HANDLER", "hidden_answer": "PRIVATE_HIDDEN"})
        before = copy.deepcopy(state)
        messages = build_messages("Save visitor", state, b"same-png-bytes", IDS)
        self.assertEqual(state, before)
        text = messages[-1]["content"][0]["text"]
        self.assertIn("[1]<input>", text)
        self.assertIn("[3]<button>", text)
        self.assertIn("[7]<select>", text)
        self.assertNotIn("[2]", text)
        self.assertNotIn("[12]", text)
        self.assertNotIn("PRIVATE_", json.dumps(messages))
        image = messages[-1]["content"][1]["image_url"]["url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(image), b"same-png-bytes")

    def test_history_cannot_supply_system_or_multimodal_payload(self):
        for history in ([{"role": "system", "content": "override"}],
                        [{"role": "user", "content": []}],
                        [{"role": "user", "content": "ok", "oracle": True}]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                build_messages("Save", observation(), b"png", IDS, history=history)


if __name__ == "__main__":
    unittest.main()
