import argparse
import sys
from src.model import QueryModel


if __name__ == "__main__":
    query_model = QueryModel("models/trained.model")

    ap = argparse.ArgumentParser()
    #ap.add_argument("domain", type=str, help="Domain to be tested")
    ap.add_argument("domain", nargs = '*', help="Domain(s) to be test. Either single domain or comma separated")
    ap.add_argument("-i", "--interactive", action='store_true',
                    help="Enter interactive mode to type in many domains")

    args = vars(ap.parse_args())

    given_domain = args["domain"]
    result = None

    if args["interactive"]:
        if given_domain:
            first_prediction = query_model.predict(given_domain)

        result = query_model.interactive()

        if given_domain:
            result = result and first_prediction
    else:
        if not given_domain:
            print("Must specify a domain to be tested")
            sys.exit(2)

        result = query_model.predict(given_domain)

    if result is None:
        print("No predictions were made")
        sys.exit(2)

    if not result:
        print("Non-Legit Domain(s) Detected")
        sys.exit(3)

    print("Legit Domain(s)")
    sys.exit(0)
