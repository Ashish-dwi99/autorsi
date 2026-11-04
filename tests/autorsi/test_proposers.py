import json
import unittest

from autorsi import EditRecipe, Goal, JSONProposer, ProposalContext, TensorEdit, parse_recipe


class ProposalTests(unittest.TestCase):
    def test_strict_json_recipe_and_stop(self):
        recipe = EditRecipe((TensorEdit("layer.weight", "scale", 0.5),), "test")
        self.assertEqual(parse_recipe(json.dumps(recipe.to_dict())), recipe)
        self.assertIsNone(parse_recipe('{"stop":true}'))

    def test_malformed_duplicate_nonfinite_or_executable_output_rejected(self):
        outputs = ["import os", "```json\n{}\n```", '{"stop":1}', '{"stop":true,"stop":true}',
                   '{"edits":[],"rationale":""}',
                   '{"edits":[{"parameter":"w","operation":"add","value":NaN}],"rationale":""}',
                   '{"edits":[{"parameter":"w","operation":"add","value":1e999}],"rationale":""}',
                   '{"edits":[{"parameter":"w","operation":"exec","value":1}],"rationale":""}',
                   '{"edits":[{"parameter":"w","operation":"scale","value":true}],"rationale":""}',
                   '{"edits":[{"parameter":"w","operation":"add","value":1,"code":"x"}],"rationale":""}']
        for output in outputs:
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_recipe(output)

    def test_json_adapter_calls_transport_once_with_declared_metadata(self):
        prompts = []
        def complete(prompt):
            prompts.append(prompt)
            return '{"stop":true}'
        context = ProposalContext(Goal("accuracy"), (), {"accuracy": 0.5}, {"accuracy": 0.6}, (),
                                  {"parameters": ("w",), "max_abs_delta": 0.1})
        self.assertIsNone(JSONProposer(complete).propose(context))
        self.assertEqual(len(prompts), 1)
        self.assertIn('"accuracy": 0.6', prompts[0])
        self.assertNotIn("payload", prompts[0])


if __name__ == "__main__":
    unittest.main()
