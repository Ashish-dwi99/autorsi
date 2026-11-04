"""Provider-neutral integration contract, exercised with an offline transport."""
import json

from autorsi import AutoRSI, Goal, JSONProposer, ScalarBackend


def complete(prompt: str) -> str:
    # Replace this function with your chosen editor LLM's configured client.
    # Return the response text. Set native request timeouts and provider budget
    # limits there; AutoRSI's deadline does not cancel remote requests.
    assert '"parameters": ["gain"]' in prompt
    return json.dumps({"edits": [{"parameter": "gain", "operation": "add", "value": 0.1}],
                       "rationale": "Offline transport fixture; no LLM call is made."})


def main():
    backend = ScalarBackend({"gain": 0.0}, editable={"gain"}, max_abs_delta=0.1)
    controller = AutoRSI(backend=backend, evaluator=lambda weights: {"score": weights["gain"]},
                        goal=Goal("score", target=0.1), proposer=JSONProposer(complete),
                        output_dir="autorsi-runs/json-editor")
    print(controller.step().to_dict())


if __name__ == "__main__":
    main()
