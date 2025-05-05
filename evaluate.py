import json
import requests

def load_test_set(path="test_set.json"):
    """Load the list of {query, relevant_urls} from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate(api_url="https://shl-assessment-recommendation-system-z06m.onrender.com/recommend", k=3):
    test_set = load_test_set()
    recall_scores = []
    ap_scores     = []

    for entry in test_set:
        query = entry["query"]
        ground_truth = set(entry["relevant_urls"])

        resp = requests.post(
            api_url,
            json={"query": query},
            timeout=30
        )
        resp.raise_for_status()
        results = resp.json()["recommended_assessments"]

        pred_urls = [item["url"] for item in results][:k]

        print("="*80)
        print(f"Query: {query!r}")
        print(f"Ground truth ({len(ground_truth)} URLs):")
        for url in ground_truth:
            print("   ✔️", url)
        print(f"\nPredicted top {k} URLs:")
        for i, url in enumerate(pred_urls, start=1):
            hit_marker = "✓" if url in ground_truth else "✗"
            print(f"   {i:2d}. {url} {hit_marker}")

        hits = sum(1 for url in pred_urls if url in ground_truth)
        recall = hits / len(ground_truth) if ground_truth else 0.0
        recall_scores.append(recall)

        hits = 0
        precision_sum = 0.0
        for i, url in enumerate(pred_urls, start=1):
            if url in ground_truth:
                hits += 1
                precision_sum += hits / i
        ap = (precision_sum / min(k, len(ground_truth))) if ground_truth else 0.0
        ap_scores.append(ap)

    mean_recall = sum(recall_scores) / len(recall_scores)
    mean_map    = sum(ap_scores)     / len(ap_scores)

    print("\n" + "="*80)
    print(f"Evaluated {len(test_set)} queries")
    print(f"Mean Recall@{k}: {mean_recall:.4f}")
    print(f"Mean MAP@{k}:    {mean_map:.4f}")

if __name__ == "__main__":
    evaluate()
