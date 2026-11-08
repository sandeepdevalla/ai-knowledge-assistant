
import json

with open("evaluation/results.json") as f:
    report = json.load(f)

for item in report["results"]:
    if item["id"] in {"Q03", "Q04", "Q14", "Q15", "Q16"}:
        print(f"\n{'=' * 70}")
        print(f"{item['id']}: {item['question']}")
        print("Expected sources:", item["expected_sources"])
        print("Retrieved passages:")
        for passage in item["retrieved"]:
            print(
                f"  {passage['document']} | page {passage['page']} "
                f"| score {passage['score']:.4f}"
            )
            print(" ", passage["text"][:250].replace("\n", " "))
        print("Generated answer:", item["answer"])
