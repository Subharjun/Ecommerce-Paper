#!/usr/bin/env python3
import sys
from authenticity import AuthenticityDetector

def main():
    print("=" * 60)
    print("🧪 Testing Fake Review Authenticity Detector (Novel #2)")
    print("=" * 60)

    detector = AuthenticityDetector()

    if len(sys.argv) > 1:
        # User provided a custom review string
        text = " ".join(sys.argv[1:])
        print(f"\nAnalyzing Custom Review:\n\"{text}\"")
        res = detector.score(text)
        print("\nResults:")
        print(f"  Score: {res['score']}/100")
        print(f"  Verdict: {res['label']} {res['icon']}")
        print("\n  Signal Breakdown:")
        for k, v in res['breakdown'].items():
            print(f"    - {k}: {v:.1f}/100")
    else:
        # Default test cases
        tests = [
            "This product is AMAZING!! BEST BUY EVER!! HIGHLY RECOMMEND TO EVERYONE!! TOP PRODUCT!!",
            "I've been using this blender for about three weeks now. The motor is powerful enough for daily smoothies, but the plastic pitcher feels a bit flimsy. Overall, decent value for $45, but wouldn't pay more.",
            "great. works fine. fast shipping.",
            "I bought this vacuum for my wife. The suction is terrible and it broke after two uses. Customer service ignored my emails. Don't buy this trash!"
        ]
        for i, t in enumerate(tests, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Review: \"{t}\"")
            res = detector.score(t)
            print(f"Score: {res['score']}/100 -> {res['label']} {res['icon']}")
            
    print("\n" + "=" * 60)
    print("💡 Tip: Try bringing your own review! Run:")
    print("   python test_authenticity.py \"Your custom text here\"")
    print("=" * 60)

if __name__ == "__main__":
    main()
