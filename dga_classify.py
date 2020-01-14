import argparse
import sys
from src.model import QueryModel


if __name__ == "__main__":
    query_model = QueryModel("models/trained.model")

    ap = argparse.ArgumentParser()
    ap.add_argument("domain", type=str, help="Domain to be tested")
    ap.add_argument("-i", "--interactive", action='store_true',
                    help="Enter interactive mode to type in many domains")

    args = vars(ap.parse_args())

    given_domain = args["domain"].strip()
    result = query_model.predict(given_domain)

    if args["interactive"]:
        result = query_model.interactive()

    if not result:
        print("Non-Legit Domain(s) Detected")
        sys.exit(3)

    print("Legit Domain(s)")
    sys.exit(0)
